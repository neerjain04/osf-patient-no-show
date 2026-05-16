# v10 — 50-Model CatBoost + LightGBM Ensemble

**Public LB: 0.78218**  
**Technique: 10-seed × 5-fold ensemble (50 models per algorithm)**

---

## What Changed

Previous best (v8) used 5-fold CB+LGBM blend. v10 extends this to 10 random seeds, running 5-fold CV for each seed — 50 CatBoost models + 50 LightGBM models = 100 total.

Lowered CatBoost learning rate from 0.034 (Optuna best) to 0.01 and raised iterations from 757 to 3000. This forces the model to take smaller, more careful steps with more trees — better generalisation on this categorical dataset.

## Parameters

**CatBoost:**
```
iterations     = 3000
learning_rate  = 0.01
depth          = 8
```

**LightGBM:**
```
n_estimators      = 1000
learning_rate     = 0.05
num_leaves        = 127
min_child_samples = 20
subsample         = 0.8
colsample_bytree  = 0.8
```

**Seeds:** `[0, 7, 42, 123, 456, 999, 1337, 2024, 31337, 77777]`

## Results

| Metric | Value |
|---|---|
| CB OOF AUC | ~0.7760 |
| LGBM OOF AUC | ~0.7735 |
| Blend OOF AUC | ~0.7770 |
| Public LB | **0.78218** |

## Key Observations

- Significant improvement over single-seed v8 (+0.00056). Multi-seed averaging reduces variance noticeably.
- CB at lr=0.01 outperformed Optuna-tuned lr=0.034 on LB — lower learning rate benefits from more iterations even if CV looks similar.
- Blend weight optimised on OOF AUC: CB weighted ~0.65, LGBM ~0.35.

## Conclusion

**v10 became the reference anchor for all future experiments.** OOF arrays saved as `cb_v10_oof.npy` and `lgbm_fe_oof.npy` and reused in greedy ensemble selection.
