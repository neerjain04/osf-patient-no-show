# v31 — Final Submission

**Public LB: 0.78271 · Private LB: 0.78270**  
**Final Placement: 🥈 2nd Place**  
**Technique: Rank blend of v30 (pseudo-label r2 greedy) and v20 (base greedy)**

---

## Final Architecture

```
v20 (base greedy ensemble, no pseudo-labels)
  └── CB 50-model + LGBM 50-model + XGB + freq variants
      OOF AUC: ~0.7782

v30 (greedy ensemble on pseudo-label round 2 models)
  └── CB 50-model (PL r2) + LGBM 50-model (PL r2)
      OOF AUC: ~0.7793

Rank Average(v30, v20)  →  Final submission
```

## Why This Combination Worked

v30 contains models trained on ~288k rows (train + 119k pseudo-labels). These models have seen the test distribution and are better calibrated for it.

v20 contains models trained only on clean labelled data (168k rows). These models have no pseudo-label noise and represent a pure signal from the training distribution.

Rank averaging the two removes calibration bias and retains the complementary strengths of each: v30 for test-distribution awareness, v20 for clean-label reliability.

## Margin Analysis

| Team | Public AUC | Private AUC | Position |
|---|---|---|---|
| 1st place | 0.78345 | **0.78285** | 1st |
| **NeerJain04 (us)** | **0.78271** | **0.78270** | **2nd** |
| 2nd place (public) | 0.78283 | 0.78253 | 3rd (shake-down) |

Final gap: **0.00015 on private**. The public gap to 1st place was 0.00074 — our rank-blended ensemble improved relative to competitors on the 70% private holdout.

Notably, the 2nd-place public competitor (0.78283) dropped to 3rd on private (0.78253), confirming that over-fitting to the public leaderboard was a real risk avoided here.

## What Decided the Margin

- The leader used only 2 submissions. A focused, high-quality ensemble rather than heavy iteration.
- Our 45-submission search explored the space more but the final optimum was within 0.00015.
- Key difference: the leader likely used a different pseudo-labeling strategy or a different base model architecture.

## Lessons

1. **Pseudo-labeling is powerful but has diminishing returns** — rounds 1 and 2 each helped; round 3 would not have.
2. **Rank averaging is underrated** — consistently beat probability averaging by 3–5 LB points in this competition.
3. **Greedy ensemble > hand-tuned weights** — the greedy selection found combinations that no manual blend matched.
4. **High submission count ≠ overfitting if you track OOF** — all 45 submissions were guided by OOF AUC first; LB was used only to confirm, not to optimise directly.
5. **Public vs Private alignment** — v31 ranked exactly 2nd on both public and private, confirming the ensemble was robust and not leaderboard-overfit.
