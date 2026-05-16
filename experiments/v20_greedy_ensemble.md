# v20 — Greedy Ensemble Selection

**Public LB: 0.78245**  
**Technique: Hill-climbing greedy ensemble over all saved OOF arrays**

---

## What Changed

Instead of hand-tuning blend weights between CB and LGBM, v20 applies greedy hill-climbing over all available OOF prediction arrays (CatBoost variants, LightGBM variants, XGBoost, frequency-encoded variants).

The algorithm:
1. Start with the single model that has the highest OOF AUC.
2. At each step, try adding each remaining model and pick the one that improves OOF AUC the most.
3. Repeat until no addition improves AUC.
4. The ensemble is the average of all selected model arrays.

## Models in the Pool

- `cb_v10` — CatBoost 50-model (lr=0.01, iter=3000)
- `lgbm_fe` — LightGBM with frequency encoding
- `cb_fe` — CatBoost with frequency encoding
- `xgb` — XGBoost ordinal-encoded
- `freq_cb` — CatBoost with additional FREQ columns
- `freq_lgbm` — LightGBM with additional FREQ columns

## Results

| Metric | Value |
|---|---|
| Greedy-selected OOF AUC | ~0.7782 |
| Public LB | **0.78245** |

## Key Observations

- Greedy selection chose CB-heavy combinations. LGBM added marginal gain only when frequency-encoded variant was included.
- XGBoost was selected in some iterations but its contribution was small.
- OOF AUC improvement from v10 to v20: ~+0.0012 on OOF, +0.00027 on LB.
- The greedy result generalised well — no sign of OOF overfitting.

## Saved Artifacts

- `cb_v10_oof.npy`, `cb_v10_test.npy`
- `lgbm_fe_oof.npy`, `lgbm_fe_test.npy`
- `xgb_oof.npy`, `xgb_test.npy`
- `freq_cb_oof.npy`, `freq_cb_test.npy`

v20 OOF/test arrays were carried forward and used as a fixed component in all future rank blends.
