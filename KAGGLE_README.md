# P-SVM vs CoRe on a 30-hour Kaggle T4 budget

This folder is CoRe's official code ([nihx9919/CoRe](https://github.com/nihx9919/CoRe)) plus what's needed to
run the comparison in *Part 1b: Kaggle Experiment Plan*. `git diff` shows every change made to CoRe's files.

**Running by hand from VS Code through a tunnel?** Follow [kaggle/VSCODE_GUIDE.md](kaggle/VSCODE_GUIDE.md).
It also covers where to get free API keys and the free-tier limits. This README covers the notebook route.

## What runs where

| Phase | What | Hardware | GPU quota |
|---|---|---|---|
| 1 | Env Dense, Env Sparse, FRM only, oracle SVM (A1–A4), plus A5/A6. State only: no images, no API | Kaggle **CPU** sessions or your Mac | none |
| 2 | CoRe re-run, RRM only, RRM + oracle SVM (B1, B2, B4), plus the B3 control. Images, VLM labels | Kaggle **GPU T4** | all of it |

Phase 2 follows CoRe's protocol exactly: 300×300 frames, 500 labels, 500k steps, CoRe's hyperparameters.
Frames are stored losslessly compressed so a run fits in Kaggle's RAM. Measured on MetaWorld frames, PNG is
2.5× smaller and decodes to identical pixels; replay/label buffers give identical results to CoRe's original
storage (tested). Expect about 23 GB peak RAM per Phase 2 run, so **one Phase 2 run per session**; the
second T4 stays idle. `check_setup.py` prints the real number for your session.

## Spending the 30 GPU-hours

Runs inside a plan are ordered seed by seed, so a plan cut short still leaves complete, comparable sets.

| Order | Session | `PLAN` | Runs | GPU-hours |
|---|---|---|---|---|
| 0 | GPU | `smoke_phase2` with `LABELS='scripted'`, then with `'vlm'` | 2 short | ~0.5 |
| 1 | CPU (free) | `phase1` (repeat sessions until all 60 runs are done) | 60 | 0 |
| 2 | GPU | `phase2_peg`: CoRe, RRM only, RRM + SVM × 3 seeds | 9 | ~3 per run (estimate) |
| 3 | GPU, if hours remain | `phase2_peg_control`: RRM + success × 3 seeds | 3 | ~3 per run |
| later | teammates' quota | `phase2_hammer`, `phase2_button`, `phase1_extra` | | |

- **Time the first full Phase 2 run before planning the rest.** If runs take ~4 h, 9 runs need 36 h.
  The launcher still stops cleanly: you get complete seeds 0–1 (6 runs), and seed 2 can go on a teammate's account.
- Every Kaggle account has its own weekly quota. Three teammates triple the budget. Attach each other's
  notebook outputs as inputs, and finished runs are skipped and merged automatically.
- Phase 1 already answers "is the success bit alone enough?" (A2 vs A4) for free. That's why the Phase 2
  control (B3) comes after the main Phase 2 runs.

## Step by step

1. **Upload the code** (on your Mac): `bash kaggle/make_upload_zip.sh` creates `../CoRe_psvm_kaggle.zip`.
   Upload it on Kaggle as a new **Dataset**.
2. **Create a notebook** from `kaggle/psvm_kaggle.ipynb` (File → Import). Add the dataset as input and turn Internet on.
3. **Secrets** (Add-ons → Secrets): `GEMINI_API_KEY` (free at aistudio.google.com/apikey). The VLM is
   `gemini-3.5-flash-lite` through Google's OpenAI-compatible endpoint. CoRe's reward-code LLM uses
   `gpt-4.1-mini` if you add an `OPENAI_API_KEY` (paid; shuts down 14 Jan 2027); otherwise the free Gemini key
   (`gemini-3.8-flash`) is used. **Keep the same VLM for every Phase 2 run.** A free key may allow only about one
   Phase 2 run per day (~1,000 VLM calls per run); see the limits in kaggle/VSCODE_GUIDE.md.
   To use other models or providers, change `VLM_MODEL` / `VLM_BASE_URL` / `LLM_MODEL` / `LLM_BASE_URL` in the notebook.
4. **Smoke test (GPU session):** `PLAN='smoke_phase2'`, `LABELS='scripted'`. Run all interactively (about
   10–20 min). It checks rendering, RAM and the whole pipeline without API cost. Repeat with `LABELS='vlm'`
   to check the API (about 120 VLM calls).
5. **Real runs:** set `PLAN`, `PARALLEL` (`2` for Phase 1 on CPU, `1` for Phase 2) and `EST_HOURS`, then
   **Save Version → Save & Run All**. That runs in the background for up to 12 h and saves `/kaggle/working/runs`.
6. **Next session:** add the previous version's output as input. Its `runs` folder is merged in and finished runs are skipped.
7. **Results:** the last cell writes `/kaggle/working/results/summary.md` (tables next to the paper's numbers,
   gap closure G, verdicts) and learning-curve plots. To merge runs from several places on your Mac:
   `python scripts/summarize.py runs_kaggle1 runs_kaggle2 runs_mac --out results`.

### Phase 1 on your Mac instead (free, about 1 h per run on the M4)

```bash
cd CoRe
python3 -m venv .venv && .venv/bin/pip install -r kaggle/requirements.txt
caffeinate -i .venv/bin/python scripts/launch.py --plan phase1 --parallel 4 --threads 2 --session_hours 1000
```

Run all of Phase 1 on one platform (all Mac or all Kaggle), so platform differences don't mix into the comparison.

## Running one experiment by hand

```bash
python train_kaggle.py --mode svm --task peg-insert-side-v2 --seed 0 --out runs            # Phase 1
python train_kaggle.py --mode rrm_svm --task peg-insert-side-v2 --seed 0 --out runs        # Phase 2
```

Modes: `env_dense env_sparse frm svm svm_sparse svm_seeded` (Phase 1), `core rrm_only rrm_sparse rrm_svm` (Phase 2).
Each run writes `runs/<task>/<mode>/seed<k>/` with `eval.jsonl` (success every 10k steps), `episodes.jsonl`,
`events.jsonl` (RRM updates, SVM switch-on, first success), `config.json` and `done.json`.

`svm_seeded` starts the discriminator with 10 successful episodes from your finished Env Dense run of the
same task (the launcher finds it). MetaWorld's scripted policies don't solve CoRe's modified Peg Insert and
Hammer scenes, and only work on Button Press.

## Rules that keep the comparison fair

- Don't change hyperparameters between variants. `--set` is for smoke tests only; `summarize.py` excludes such runs.
- One VLM model ID for all of Phase 2 (`summarize.py` warns if it sees more than one).
- `--labels scripted` (true preferences instead of the VLM) is for testing only. Those runs aren't comparable to the paper.
- Use the bundled MetaWorld, never `pip install metaworld`.
- A run continued after an early stop (`HOURS=...`) restarts its replay buffer empty at that step; `summarize.py` lists such runs. Report it.
- `rrm_psvm` (B5) asks the VLM one extra question per episode (about 1,000 per run: how far the final frame got, 0-10) on top of the 500 preference queries. Report that budget next to the other rows.

## What changed in CoRe's files

| File | Change | Effect on results |
|---|---|---|
| `replay_buffer.py` | frames go through `psvm/frames.py`; per-step success column; `relabel_structural`; `drop_images` | none (tested identical) |
| `RRM/reward_model.py` | lossless frame storage; optional scripted labels; parallel VLM queries; label files saved without images; fixed the logged VLM-accuracy formula (logging only) | none |
| `RRM/preference_label.py`, `FRM/code_agent.py` | model, key and endpoint read from environment variables (`VLM_*`, `LLM_*`); optional query videos | none |
| `FRM/reward_code.py` | can skip editing MetaWorld source files at run time | none |
| `utils.py` | moviepy / scikit-image imports made optional | none |
| new: `train_kaggle.py`, `psvm/`, `scripts/`, `kaggle/` | the experiment | |

`train_kaggle.py` follows `train_CoRe.py` step for step: same warm-up, RRM schedule, critic reset,
relabeling, evaluation and FRM refinement. It differs only in which reward fills the FRM slot. It also stops
storing replay images after the last RRM relabel (~116k steps), because nothing reads them afterwards.

## Tested and not yet tested

Tested on a Mac (CPU):
- every Phase 1 mode;
- Phase 2 `rrm_svm` and `core` end to end with scripted labels;
- the launcher, the summary and the setup check;
- storage equivalence (ram / png / disk).

Not tested (no GPU or API keys here):
- EGL rendering on Kaggle;
- real VLM/LLM calls, including CoRe's LLM refinement step in `core` runs;
- full-length runs and their wall-clock time on a T4.

The smoke tests in step 4 cover the first two. Time your first full run before planning the rest.

## If something goes wrong

- **Killed / out of memory in Phase 2:** use `PARALLEL=1`. If it still fails, run with `--extra --frame_backend disk`
  (frames on disk) if `check_setup.py` shows enough free disk.
- **Rendering error (EGL):** re-run `kaggle/setup.sh` (it installs `libegl1`) and make sure the session has a GPU.
- **`No API key for the VLM`:** the Secret isn't attached to this notebook, or it has a different name.
- **VLM errors or rate limits:** lower `--vlm_workers` with `--extra --vlm_workers 2`. Failed queries are retried
  7 times, then skipped as in the original code.
