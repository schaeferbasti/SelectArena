# calculco_selectarena_run vs. calculco

- Matched runs: 456 (450 with feature selection); missing in calculco: 0, missing in calculco_selectarena_run: 0.
- Same budget (max_features): 100.0% of the runs.
- Identical selected features: 96.4% of the runs (mean Jaccard 0.984).
- Error tolerance (from LightGBM on all features): 1.0% relative difference.
- Leaderboard: Spearman correlation of the average ranks 0.989.

## Per method

| Method | Runs | Identical selection | Mean Jaccard | Median / max rel. error diff | Fallback runs calculco_selectarena_run / calculco | Median FS time calculco_selectarena_run / calculco (s) | Check |
|---|---|---|---|---|---|---|---|
| (R)ReliefF | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 42.7 / 43.2 | ok |
| All features (default) | 6 | n/a | n/a | 0.0% / 0.0% | 0 / 0 | n/a / n/a | ok |
| CART | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.0 / 0.1 | ok |
| ElasticNet | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 10.8 / 9.7 | ok |
| F-Test | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.0 / 0.0 | ok |
| GainRatio | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.1 / 0.1 | ok |
| LOCO | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 119.8 / 128.3 | ok |
| LaplacianScore | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.1 / 0.1 | ok |
| Lasso | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 2.3 / 2.0 | ok |
| MI | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.2 / 0.2 | ok |
| MarkovBlanket | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.1 / 0.1 | ok |
| RFImportance | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.2 / 0.3 | ok |
| Random | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 30 / 30 | 0.0 / 0.0 | ok |
| SBE | 30 | 80.0% | 0.882 | 0.0% / 12.4% | 6 / 4 | 611.8 / 642.4 | selections differ (mean Jaccard 0.88); fallback differs |
| SFS | 30 | 66.7% | 0.886 | 0.0% / 7.9% | 11 / 11 | 937.0 / 983.7 | selections differ (mean Jaccard 0.89) |
| mRMR | 30 | 100.0% | 1.000 | 0.0% / 0.0% | 0 / 0 | 0.3 / 0.3 | ok |

## Leaderboard (all datasets, splits, budgets)

| Method | Avg. rank calculco_selectarena_run | Avg. rank calculco | Win rate calculco_selectarena_run | Win rate calculco |
|---|---|---|---|---|
| Lasso | 4.62 | 4.62 | 0.742 | 0.742 |
| ElasticNet | 4.87 | 4.83 | 0.724 | 0.726 |
| F-Test | 5.10 | 5.13 | 0.707 | 0.705 |
| MI | 5.68 | 5.68 | 0.665 | 0.665 |
| RFImportance | 5.72 | 5.72 | 0.663 | 0.663 |
| SBE | 5.98 | 5.48 | 0.644 | 0.680 |
| CART | 6.42 | 6.42 | 0.613 | 0.613 |
| LOCO | 6.87 | 6.90 | 0.581 | 0.579 |
| (R)ReliefF | 7.20 | 7.20 | 0.557 | 0.557 |
| SFS | 9.22 | 9.78 | 0.413 | 0.373 |
| mRMR | 10.77 | 10.73 | 0.302 | 0.305 |
| MarkovBlanket | 11.17 | 11.13 | 0.274 | 0.276 |
| GainRatio | 11.58 | 11.58 | 0.244 | 0.244 |
| Random | 11.98 | 12.02 | 0.215 | 0.213 |
| LaplacianScore | 12.83 | 12.77 | 0.155 | 0.160 |
