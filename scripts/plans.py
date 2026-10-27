"""Run lists for the Kaggle experiment plan (Part 1b).

Runs are ordered seed-first, so a plan stopped part-way still has complete,
comparable sets of variants for the seeds that did finish.
"""

HARD_TASKS = ['peg-insert-side-v2', 'hammer-v2', 'button-press-topdown-v2']

PHASE1_MAIN = ['env_dense', 'env_sparse', 'frm', 'svm']          # A1-A4
PHASE1_EXTRA = ['svm_sparse', 'svm_seeded']                       # A5, A6
# proposal v2: CoRe's LLM reward + SVM instead of the VLM part (A7), and SVM given demonstrations (A6)
V2_RUNS = [('frm_svm', t) for t in HARD_TASKS] + [('svm_seeded', 'peg-insert-side-v2'), ('svm_seeded', 'button-press-topdown-v2')]
PHASE2_MAIN = ['rrm_svm', 'rrm_only', 'core']                     # B4 (the proposal) first, then B2, B1
PHASE2_CONTROL = ['rrm_sparse']                                   # B3


def runs(modes, tasks, seeds):
    return [(m, t, s) for s in seeds for t in tasks for m in modes]


PLANS = {
    # Phase 1: state-only, CPU is enough (no GPU quota, no API)
    'phase1': runs(PHASE1_MAIN, HARD_TASKS, range(5)),
    'phase1_extra': runs(PHASE1_EXTRA, HARD_TASKS, range(5)),
    'v2': [(m, t, s) for s in range(3) for m, t in V2_RUNS],
    # control for v2: the LLM reward + the success bit itself, no SVM
    'control': runs(['frm_sparse'], HARD_TASKS, range(3)),
    # v2 without ground truth: FRM + SVM, success labels from the VLM (Gemini) instead of the simulator
    'gt_free': runs(['frm_psvm'], HARD_TASKS, range(1)),
    # CoRe re-runs to compare v2 with in our own setup (same VLM, same code): Hammer and Button, seed by seed
    'core_more': runs(['core'], ['hammer-v2', 'button-press-topdown-v2'], range(3)),
    # Phase 2: one task at a time; the GPU budget decides how far you get
    'phase2_peg': runs(PHASE2_MAIN, ['peg-insert-side-v2'], range(3)),
    'phase2_peg_control': runs(PHASE2_CONTROL, ['peg-insert-side-v2'], range(3)),
    'phase2_hammer': runs(PHASE2_MAIN, ['hammer-v2'], range(3)),
    'phase2_button': runs(PHASE2_MAIN, ['button-press-topdown-v2'], range(3)),
    # P-SVM with VLM-graded soft success weights (the fix for the cold start on Peg Insert)
    'psvm_peg': runs(['rrm_psvm'], ['peg-insert-side-v2'], range(3)),
    'psvm_hammer': runs(['rrm_psvm'], ['hammer-v2'], range(3)),
    # a few minutes each: checks the whole pipeline before spending quota
    'smoke_phase1': runs(['env_dense', 'svm'], ['button-press-topdown-v2'], [0]),
    'smoke_phase2': runs(['rrm_svm'], ['button-press-topdown-v2'], [0]),
}

SMOKE_SETTINGS = {
    'smoke_phase1': ['num_train_steps=3000', 'num_seed_steps=500', 'eval_frequency=1000',
                     'num_eval_episodes=2', 'svm_min_effective=1'],
    'smoke_phase2': ['num_train_steps=6000', 'num_seed_steps=500', 'num_pre_steps=1500',
                     'num_interact=1000', 'max_feedback=60', 'eval_frequency=3000',
                     'num_eval_episodes=2', 'svm_min_effective=1'],
}
