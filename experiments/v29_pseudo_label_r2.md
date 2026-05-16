# v29 — Pseudo-Labeling Round 2

**Public LB: 0.78269**  
**Technique: Second round of pseudo-labeling with stricter thresholds**

---

## What Changed from Round 1

Round 1 (v22) used the v10 50-model ensemble to generate pseudo-labels. Round 2 uses the **v23 greedy ensemble** (which already incorporates round 1 pseudo-labels) as a stronger oracle for generating new labels.

Stricter thresholds were also applied to maintain label quality despite the larger pseudo-label pool.

## Procedure

1. Generate test predictions using the v23 greedy ensemble (best model at the time).
2. Apply stricter confidence thresholds:
   - Predict **no-show** (label=1) if probability > **0.75** (up from 0.70)
   - Predict **showed up** (label=0) if probability < **0.06** (down from 0.08)
3. Add ~119,000 pseudo-labeled rows to the original training set.
4. Retrain CatBoost + LightGBM 50-model ensembles on the augmented dataset.
5. Greedy ensemble selection → v30.
6. Rank blend v30 with v20 → v31.

## Results

| Metric | Value |
|---|---|
| Pseudo-labeled rows added | ~119,000 |
| New training set size | ~288,000 rows |
| OOF AUC | ~0.7793 |
| Public LB (v29 direct) | **0.78269** |

## Key Observations

- Marginal improvement over round 1 (+0.00019 from v22). Diminishing returns — the model was already well-calibrated.
- The stricter thresholds were essential. Looser thresholds at round 2 hurt LB despite higher raw row count.
- Saved as `cb_pl2_oof.npy`, `lgbm_pl2_oof.npy`.

## What Didn't Work

- A third pseudo-label round: OOF AUC stopped improving. The model's pseudo-labels were increasingly identical to its own predictions — no new information.
- Iterative pseudo-labeling (re-label after every epoch): overfitting to test noise.
