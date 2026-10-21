# Running the experiments from VS Code on a Kaggle machine

You open a VS Code tunnel from a Kaggle notebook, connect to it from VS Code on your laptop, and run
everything by hand in the VS Code terminal. Your code, runs and logs live on the Kaggle machine; the
Kaggle browser tab just keeps the session alive.

---

## 0. Accounts and keys (all free)

| What | Where to get it | Used for |
|---|---|---|
| Kaggle account, **phone-verified** | kaggle.com → Settings → Phone verification | GPU quota (~30 h/week) and Internet in notebooks |
| GitHub account | github.com | Signing in to the VS Code tunnel |
| **Gemini API key** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → *Create API key* (sign in with a Google account; no card needed) | VLM preference labels (Phase 2), and CoRe's reward-code LLM |
| Kaggle API token (optional) | kaggle.com → Settings → API → *Create New Token* (downloads `kaggle.json` with your username and key) | Saving and loading results as a private Kaggle dataset from the terminal |

**About the free Gemini tier:**

- Google doesn't publish fixed free limits. Your own limits are shown at
  [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit). Third-party guides report roughly
  15–30 requests/minute and around 1,000 requests/day for Flash-Lite. Check yours.
- **One Phase 2 run makes about 1,000 VLM calls** (500 labels × a 2-step prompt). With a ~1,000/day limit,
  that's **about one Phase 2 run per day per key**. The daily count resets at midnight Pacific time.
- Set `VLM_RPM` in your keys file to your per-minute limit. The code then paces its requests, and backs off for
  up to ~30 min on "too many requests" errors before giving up.
- On the free tier Google may use your prompts to improve its products. Here those are simulator images, so
  that's fine; don't send anything private.
- OpenAI's API has no free tier, so the LLM role (CoRe's reward-code refinement, ~16 calls per `core` run) uses
  the same Gemini key (`gemini-3.8-flash` instead of the paper's `gpt-4.1-mini`). **Say this in your report.**

---

## 1. The code lives on GitHub

Our version of CoRe is the branch `psvm-kaggle` of your fork, `github.com/kanu2704/CoRe`.
Kaggle downloads it with `git clone`, so nothing large goes through the tunnel. After changing code on your
Mac, `git push`, then `git pull` on Kaggle. (The old upload-a-zip route via `kaggle/make_upload_zip.sh`
still works as a fallback.)

## 2. Start the tunnel (each session, in the browser)

1. First time only: **Code → New Notebook → File → Import Notebook →** `kaggle/vscode_tunnel.ipynb`. Reuse it afterwards.
2. In the notebook's right-hand panel:
   - **Accelerator:** *GPU T4 x2* for Phase 2, *None* for Phase 1.
   - **Internet:** On.
   - **Add Data → Your Datasets →** your `psvm-runs` results (once you have some).
3. Run the code cell. It prints a `github.com/login/device` link and a code. Open the link, enter the code, approve.
4. **Leave the cell running and the tab open.** Stopping the cell closes the tunnel, and the session ends after 12 h at most.

## 3. Connect VS Code (on your laptop)

1. Install the extension **Remote - Tunnels** (by Microsoft).
2. Press Cmd+Shift+P → **Remote-Tunnels: Connect to Tunnel…** → sign in with the same GitHub account → choose **kaggle-psvm**.
3. **Terminal → New Terminal**. This terminal runs on the Kaggle machine.
4. Optional: **File → Open Folder →** `/kaggle/working` to see files and download results.

## 4. Set up the session (VS Code terminal, every session)

```bash
cd /kaggle/working && git clone --depth 1 -b psvm-kaggle https://github.com/kanu2704/CoRe.git
```

```bash
bash /kaggle/working/CoRe/kaggle/start.sh
```

This builds the pinned Python environment (~3 min) and merges any earlier results you attached.

**Your keys.** The Kaggle machine is wiped when a session ends, so add them each session:

```bash
code ~/.psvm_keys.env
```

Paste in `VLM_API_KEY` and `LLM_API_KEY` (the same Gemini key for both), `VLM_RPM` (your limit), and optionally
`KAGGLE_USERNAME` / `KAGGLE_KEY`. Then save the file. Two tips:

- Keep a filled copy of this file on your laptop and paste it in each time.
- Instead, you can attach your Gemini key as a Kaggle Secret named `GEMINI_API_KEY` (Add-ons → Secrets in the
  tunnel notebook). `env.sh` tries to read it if the file has no key.

Then:

```bash
cd /kaggle/working/CoRe && source kaggle/env.sh
```

```bash
$PY scripts/check_setup.py --render --api
```

The check prints: the GPU; RAM and disk; whether MuJoCo rendering works; how much the frames compress; how many
Phase 2 runs fit in RAM at once; and whether the VLM and LLM keys answer.

## 5. Run (VS Code terminal)

Every run starts in the background. It keeps going if VS Code disconnects, but **stops when the Kaggle session
ends**, so the launcher only starts runs that can finish in the time left.

**First GPU session: smoke tests** (about 15 min each on a T4):

```bash
bash kaggle/run.sh smoke_phase2 --labels scripted
```

```bash
bash kaggle/run.sh smoke_phase2
```

The first is free (no API calls). The second uses your Gemini key for about 120 calls. Smoke runs go to
`runs/_smoke/` and never count as results.

**Watching progress:**

```bash
$PY scripts/status.py /kaggle/working/runs
```

```bash
tail -f /kaggle/working/runs/launch_*.log
```

**Real runs:**

| Session | Command | Notes |
|---|---|---|
| CPU (Accelerator *None*) | `bash kaggle/run.sh phase1` | 60 state-only runs, 2 at a time, no GPU quota or API used. Or run Phase 1 on your Mac (see `KAGGLE_README.md`) |
| GPU T4 x2 | `bash kaggle/run.sh phase2_peg --max_runs 1` | CoRe, RRM-only and RRM + SVM on Peg Insert, 3 seeds each (9 runs). `--max_runs 1` = one run per session/day for a free-tier key |
| GPU T4 x2 | `bash kaggle/run.sh phase2_peg_control --max_runs 1` | The success-bit control, if quota remains |

More launcher options:

- `--est_hours 4` changes the time estimate used before the first run has been timed (the default is 3 h).
- `--only seed=0` picks particular runs.
- Runs are done seed by seed, so stopping early still leaves complete sets for comparison.

## Labels from a free Qwen model (default on GPU T4 x2)

Gemini's free tier allows 500 Flash-Lite requests per day, and one Phase 2 run needs ~1,000. So on a
two-GPU session `go.sh` first starts **Qwen3-VL-4B-Instruct** from Hugging Face on the second T4
(`kaggle/start_vlm.sh`, logs in `/tmp/vlm_server_<gpu>.log`), and the image reward model's labels come from it:
free, no daily limit. CoRe's reward-code rewriting (~16 calls per `core` run) still uses Gemini Flash-Lite.
Use the same labelling model for every Phase 2 run. To use Gemini for labels instead: `PSVM_VLM=gemini bash kaggle/go.sh`.

## Stopping early and continuing later

`HOURS=2.5 bash kaggle/go.sh` stops the run itself about 15 min before that many hours. A run stopped this way
(or by the VLM's daily quota) saves its networks, optimizers and labelled frames, and the next `bash kaggle/go.sh`
continues it from that step: later the same week, next week, or in another account. What isn't kept is the replay
buffer: the continued run refills it for 1,000 steps before SAC trains again, and the results table marks the run
as continued. A run cut off by the session itself ending (no `incomplete.json`) can't continue and starts again
from step 0.

- **Same account:** `/kaggle/working` is kept between sessions, so just run `bash kaggle/go.sh` again.
- **Another account:** `bash kaggle/save_results.sh`, download the zip, then in the other account's tunnel notebook
  **Add Data → Upload** the zip before pressing Run. `go.sh` copies the stopped run in and continues it.

## 6. Save results before the session ends

```bash
bash kaggle/save_results.sh
```

This writes the tables and plots to `/kaggle/working/results/summary.md` and packages everything as
`/kaggle/working/psvm_runs_<time>.zip`. In VS Code's Explorer, **right-click the zip → Download**.

With a Kaggle API token you can save it to Kaggle instead:

```bash
bash kaggle/save_results.sh --upload
```

That creates or updates your private dataset `<username>/psvm-runs`. In the next session, attach it with
**Add Data** before starting the tunnel, and `start.sh` merges it automatically. Finished runs are then skipped.

To combine downloads from several sessions or teammates on your Mac:

```bash
python scripts/merge_runs.py psvm_runs_*.zip --into runs && python scripts/summarize.py runs --out results
```

## 7. Plan for the budgets

| Budget | Limit | What fits |
|---|---|---|
| Kaggle GPU | ~30 h/week | ~7–9 Phase 2 runs of ~3–4 h (time your first one) |
| Gemini free key | ~1,000 requests/day (check yours) | ~1 Phase 2 run/day |
| Kaggle CPU / your Mac | no quota | all of Phase 1 |

So a realistic week is Phase 1 on CPU (or your Mac), plus one Phase 2 run per day on the GPU. Teammates with
their own Kaggle and Google accounts can run their share of the Phase 2 runs in parallel and attach each
other's `psvm-runs` results.

## 8. If something goes wrong

| Symptom | Fix |
|---|---|
| `git clone` fails | Check Internet is On in the notebook settings |
| `VLM ... (key MISSING)` | Fill `~/.psvm_keys.env`, then `source kaggle/env.sh` again |
| Logs show 429 / "quota" errors | Normal pacing and backoff. If a run ends with "too many failed attempts", the daily quota is used up: wait for the reset |
| Run killed, no error | Out of RAM: one Phase 2 run at a time (the default) |
| Rendering / EGL error | Make sure the session has a GPU; re-run `bash kaggle/setup.sh` |
| Tunnel name already in use | Change `--name kaggle-psvm` in the notebook cell |
