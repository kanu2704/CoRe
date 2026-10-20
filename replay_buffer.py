import numpy as np
import torch
from psvm.frames import FrameArray

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, image_size=300, reward_alpha=0.5,
                 image_backend='ram', png_level=1):
        self.capacity = capacity
        self.device = device
        # for time decrease
        self.reward_alpha = reward_alpha

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)

        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards_frm = np.empty((capacity, 1), np.float32)
        self.rewards_rrm = np.empty((capacity, 1), np.float32)
        # per-step simulator success flag (used by success-based structural rewards)
        self.successes = np.zeros((capacity, 1), np.float32)

        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.images = None
        if image_size:
            # lossless storage: 'ram' is the original numpy array, 'png'/'disk' save RAM
            self.images = FrameArray(capacity, image_size, image_size, backend=image_backend,
                                     png_level=png_level, name='replay')

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, reward_frm, reward_rrm,
            next_obs, done, done_no_max, image=None, success=0.0):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)

        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.rewards_frm[self.idx], reward_frm)
        np.copyto(self.rewards_rrm[self.idx], reward_rrm)
        self.successes[self.idx] = success

        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if image is not None and self.images is not None:
            self.images.put(self.idx, image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def drop_images(self):
        """Free stored frames once the image reward model stops relabeling; nothing reads them afterwards."""
        if self.images is not None:
            self.images.release()
            self.images = None

    def relabel_structural(self, reward_fn):
        """Recompute the structural (FRM-slot) reward of every stored transition with a
        vectorised reward_fn(obses, actions, successes) and refresh the total reward."""
        n = self.idx if not self.full else self.capacity
        if n == 0:
            return
        self.rewards_frm[:n, 0] = reward_fn(self.obses[:n], self.actions[:n], self.successes[:n, 0])
        self.rewards[:n] = (1.0 - self.reward_alpha) * self.rewards_frm[:n] + self.reward_alpha * self.rewards_rrm[:n]

    def relabel_with_predictor(self, predictor, step):

        batch_size = 32 
        relabel_max_idx = self.idx if not self.full else self.capacity
        total_iter = int(np.ceil(relabel_max_idx/batch_size))
   
        # frames are fetched (and decoded, for compressed storage) in large chunks;
        # the reward model still sees batches of 32 exactly as in the original code
        chunk_start, chunk = -1, None
        chunk_size = batch_size * 32
        for index in range(total_iter):
            start_index = index * batch_size
            last_index = (index + 1)*batch_size
            if last_index > relabel_max_idx:
                last_index = relabel_max_idx

            if chunk is None or start_index >= chunk_start + len(chunk):
                chunk_start = start_index
                chunk = self.images.take(np.arange(start_index, min(start_index + chunk_size, relabel_max_idx)))
            inputs = chunk[start_index - chunk_start:last_index - chunk_start]
            inputs = np.transpose(inputs, (0, 3, 1, 2))
            inputs = inputs.astype(np.float32) / 255.0

            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards_rrm[start_index:last_index] = pred_reward
            self.rewards[start_index:last_index] = (1.0 - self.reward_alpha) * self.rewards_frm[start_index:last_index] + self.reward_alpha * self.rewards_rrm[start_index:last_index]
        torch.cuda.empty_cache()
    
    def relabel_with_rf(self, frm, target_pos, step):
        relabel_max_idx = self.idx if not self.full else self.capacity
        for idx in range(relabel_max_idx):
            self.rewards_frm[idx], _ = frm(self.obses[idx], self.actions[idx], target_pos)
            self.rewards[idx] = (1.0 - self.reward_alpha) * self.rewards_frm[idx] + self.reward_alpha * self.rewards_rrm[idx]
    
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max
