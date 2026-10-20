"""Success Visitation Matching (SVM) reward with soft success weights.

Every finished episode i gets a success weight q_i in [0, 1]:
  * oracle labels (initial experiment): q_i = 1 if the simulator says the
    episode succeeded, else 0;
  * P-SVM later: q_i = share of YES answers from K paraphrased VLM questions.

A discriminator D(s, a) learns to tell state-action pairs of success-weighted
episodes from those of failure-weighted ones. Positives are drawn from episodes
with probability proportional to q_i and negatives proportional to 1 - q_i.
That matches the weighted cross-entropy of the plan in expectation and keeps
both halves of every minibatch the same size even when successes are rare.

    r_svm(s, a) = lambda * clip(logit D(s, a), -beta, beta)

The reward stays 0 until sum(q) >= M and sum(1 - q) >= M.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodeStore:
    """All (state, action) rows of finished episodes plus one weight per episode."""

    def __init__(self, dim, capacity_rows=65536):
        self.dim = dim
        self.data = np.empty((capacity_rows, dim), dtype=np.float32)
        self.n_rows = 0
        self.starts, self.lengths, self.q, self.y = [], [], [], []

    def __len__(self):
        return len(self.q)

    def add(self, sa, q, y=None):
        sa = np.asarray(sa, dtype=np.float32).reshape(-1, self.dim)
        need = self.n_rows + len(sa)
        if need > len(self.data):
            grown = np.empty((max(need, 2 * len(self.data)), self.dim), dtype=np.float32)
            grown[:self.n_rows] = self.data[:self.n_rows]
            self.data = grown
        self.data[self.n_rows:need] = sa
        self.starts.append(self.n_rows)
        self.lengths.append(len(sa))
        self.q.append(float(q))
        self.y.append(np.nan if y is None else float(y))
        self.n_rows = need

    def weight_sums(self):
        q = np.asarray(self.q, dtype=np.float64)
        return float(q.sum()), float((1.0 - q).sum())

    def sample(self, n, positive, rng):
        q = np.asarray(self.q, dtype=np.float64)
        w = q if positive else 1.0 - q
        ep = rng.choice(len(q), size=n, p=w / w.sum())
        lengths = np.asarray(self.lengths)[ep]
        t = (rng.random(n) * lengths).astype(np.int64)
        rows = np.asarray(self.starts)[ep] + t
        return self.data[rows]

    def stats(self):
        x = self.data[:self.n_rows]
        return x.mean(0), x.std(0) + 1e-6


class Discriminator(nn.Module):
    def __init__(self, dim, hidden=256, depth=2):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SVMReward:
    def __init__(self, obs_dim, act_dim, device='cpu', hidden=256, depth=2, lr=3e-4,
                 batch_size=512, steps_per_update=50, min_effective=5.0,
                 clip_beta=1.0, scale=1.0, weight_decay=0.0, seed=0):
        self.dim = obs_dim + act_dim
        self.device = torch.device(device)
        self.store = EpisodeStore(self.dim)
        self.disc = Discriminator(self.dim, hidden, depth).to(self.device)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.steps_per_update = steps_per_update
        self.min_effective = min_effective
        self.clip_beta = clip_beta
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self.ready = False
        self.mean = torch.zeros(self.dim, device=self.device)
        self.std = torch.ones(self.dim, device=self.device)
        self.last = {}

    def add_episode(self, obs, act, q, y=None):
        sa = np.concatenate([np.asarray(obs, np.float32), np.asarray(act, np.float32)], axis=-1)
        self.store.add(sa, q, y)

    def update(self):
        """Switch on once there is enough weight on both sides, then train.
        Returns True if the discriminator changed."""
        pos_w, neg_w = self.store.weight_sums()
        if not self.ready and (pos_w < self.min_effective or neg_w < self.min_effective):
            return False
        self.ready = True
        mean, std = self.store.stats()
        self.mean = torch.as_tensor(mean, device=self.device)
        self.std = torch.as_tensor(std, device=self.device)
        half = self.batch_size // 2
        self.disc.train()
        for _ in range(self.steps_per_update):
            pos = torch.as_tensor(self.store.sample(half, True, self.rng), device=self.device)
            neg = torch.as_tensor(self.store.sample(half, False, self.rng), device=self.device)
            logits = self.disc((torch.cat([pos, neg]) - self.mean) / self.std)
            target = torch.cat([torch.ones(half, device=self.device), torch.zeros(half, device=self.device)])
            loss = F.binary_cross_entropy_with_logits(logits, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        with torch.no_grad():
            acc = ((logits > 0).float() == target).float().mean().item()
        self.last = {'svm_loss': loss.item(), 'svm_batch_acc': acc,
                     'svm_pos_weight': pos_w, 'svm_neg_weight': neg_w}
        return True

    @torch.no_grad()
    def logits(self, sa):
        self.disc.eval()
        x = torch.as_tensor(np.asarray(sa, np.float32), device=self.device)
        return self.disc((x - self.mean) / self.std).cpu().numpy()

    def reward_batch(self, obs, act, chunk=65536):
        obs = np.asarray(obs, np.float32)
        act = np.asarray(act, np.float32)
        if not self.ready:
            return np.zeros(len(obs), dtype=np.float32)
        out = np.empty(len(obs), dtype=np.float32)
        for s in range(0, len(obs), chunk):
            sa = np.concatenate([obs[s:s + chunk], act[s:s + chunk]], axis=-1)
            out[s:s + chunk] = self.scale * np.clip(self.logits(sa), -self.clip_beta, self.clip_beta)
        return out

    def reward(self, obs, act):
        return float(self.reward_batch(np.asarray(obs)[None], np.asarray(act)[None])[0])

    def save(self, path):
        torch.save({'disc': self.disc.state_dict(), 'mean': self.mean, 'std': self.std,
                    'ready': self.ready}, path)
