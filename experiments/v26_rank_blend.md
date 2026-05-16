# v26 — Rank Blend of v23 and v20

**Public LB: 0.78268**  
**Technique: Rank averaging across two independent greedy ensembles**

---

## What Is Rank Averaging?

Different models output probabilities on different scales. CatBoost may output 0.12 for a borderline case while LightGBM outputs 0.08 for the same row — not because one is more confident, but because they have different calibration.

Rank averaging converts each model's probabilities to **percentile ranks** (0 to 1) before averaging. This removes calibration differences and focuses purely on the relative ordering of predictions.

```python
def rank_avg(preds_list):
    ranks = [pd.Series(p).rank(pct=True).values for p in preds_list]
    return np.mean(ranks, axis=0)
```

## Components

| Component | Description | OOF AUC |
|---|---|---|
| v23 | Greedy ensemble on pseudo-label r1 models | ~0.7790 |
| v20 | Greedy ensemble on base models (no pseudo-labels) | ~0.7782 |

v23 and v20 were trained on different data distributions (v23 saw pseudo-labeled rows). This makes them complementary — their errors are partially independent.

## Why Rank Over Probability?

- v23 predictions are calibrated toward the augmented (pseudo-labeled) distribution.
- v20 predictions are calibrated toward the clean training distribution.
- Direct probability averaging would be biased toward whichever model has a higher raw mean.
- Rank averaging normalises both to the same scale before combining, consistently outperforming probability averaging on this dataset.

## Results

| Method | Public LB |
|---|---|
| v23 alone | 0.78260 |
| v20 alone | 0.78245 |
| Prob avg (v23, v20) | 0.78263 |
| **Rank avg (v23, v20)** | **0.78268** |

## Conclusion

Rank averaging outperformed probability averaging by 0.00005. Held as the best submission until v29/v31 extended this with round 2 pseudo-labels.
