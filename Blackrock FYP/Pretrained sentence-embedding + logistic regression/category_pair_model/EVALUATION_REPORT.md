# Category/subcategory pair model — grouped evaluation

Dataset: 2,625 eligible tickets, 39 pairs, 803 connected groups. Eleven rows lack usable text or both usable labels and are excluded (see excluded_rows.csv). The cohort is independent of escalation eligibility: missing escalation outcomes do not prevent category training.

| Metric | Embeddings + LR | Training-majority pair |
|---|---:|---:|
| Exact pair accuracy | 99.92% | 10.32% |
| Category accuracy | 100.00% | 26.90% |
| Subcategory accuracy | 99.92% | 10.32% |
| Pair macro F1 | 99.95% | 0.48% |

Five outer grouped folds yield one held-out prediction per ticket. Three inner grouped folds choose C by pooled pair macro F1. Both errors were Investor Portal tickets whose true subcategory was Document not visible and predicted subcategory was Navigation question. IDs: TKT0100292, TKT0100889.

The corpus is synthetic and templated. Exact duplicate text and connected incident/problem records stay in one fold, but similar templates can still occur across groups. The very high score does not show real-world accuracy or superiority to the existing routing system. Three pairs each have only three independent groups, so minority-class estimates are fragile. Rare inner training classes can trigger stratification warnings. Unseen test pairs are retained in evaluation; the classifier cannot predict a pair absent from training.

The separately saved final model is trained on all eligible rows after grouped hyperparameter selection. It is not used to calculate these held-out results. All outputs are text-only; category/subcategory labels never enter the encoder as structured features.
