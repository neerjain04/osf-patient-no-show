# v22 — Pseudo-Labeling Round 1

**Public LB: 0.78250**  
**Technique: High-confidence test rows added as training data**

---

## What Is Pseudo-Labeling?

The test set has no labels, but after training a strong ensemble we can assign "soft" labels to test rows the model is most confident about. These high-confidence rows are added to the training set, effectively increasing training data size and letting the model learn from patterns in the test distribution.

The risk is noise: wrong pseudo-labels hurt more than they help. We mitigate this by using very strict confidence thresholds.

## Procedure

1. Generate test predictions using the v10 50-model ensemble.
2. Apply confidence thresholds:
   - Predict **no-show** (label=1) if probability > **0.70**
   - Predict **showed up** (label=0) if probability < **0.08**
   - Discard rows in the uncertain middle band.
3. Add ~95,000 pseudo-labeled test rows to the training set.
4. Retrain CatBoost 50-model ensemble on the augmented dataset.
5. Greedy ensemble selection on new OOF arrays.

## Results

| Metric | Value |
|---|---|
| Pseudo-labeled rows added | ~95,000 |
| New training set size | ~264,000 rows |
| OOF AUC (on original train) | ~0.7785 |
| Public LB | **0.78250** |

## Key Observations

- +0.00032 LB improvement over v10 — the largest single jump in the competition.
- The improvement came almost entirely from the negative class (showed-up rows): ~90k of the 95k pseudo rows were label=0. This gave the model a clearer picture of the majority class distribution in the test set.
- Pseudo-labeled models saved as `cb_pl_oof.npy`, `lgbm_pl_oof.npy`.

## What Didn't Work

- Threshold 0.60/0.10 (more rows, looser): hurt LB by ~0.00010. More noise than signal.
- Adding pseudo-labels to LightGBM only (not CatBoost): smaller gain.
- Using pseudo-labels in XGBoost: marginal, not worth the extra training time.
