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

# Added semantic template-family proxy experiment

Cosine threshold 0.90 was fixed before evaluation; no threshold search based on model accuracy was performed. Links between highly similar descriptions are combined with original groups by transitive connected components.

| Metric | Original grouping | Semantic family proxy |
|---|---:|---:|
| Groups | 803 | 103 |
| Pair accuracy | 99.92% | 77.71% |
| Pair macro F1 | 99.95% | 73.15% |
| Category accuracy | 100.00% | 80.11% |
| Subcategory accuracy | 99.92% | 77.71% |

IMPORTANT: one connected component contains 2,063 tickets (78.59%). Outer test fold sizes are 212, 109, 108, 2,088, 108. The large test fold has only 537 training tickets and 254 test tickets whose true pair is absent from training. These cases remain in the score, rather than being silently excluded. This is a highly imbalanced exploratory stress test, not a validated estimate of performance on new template families. The decline includes both different descriptions and insufficient training-label coverage.

Review the label-blind threshold examples and individual similarity links, especially bridges joining distinct issue types. Prefer verified generator template IDs when possible. Do not break incident/problem links just to improve the score, and do not select a cutoff by the reported test results. Retain this experiment as exploratory if you adjust the rule after seeing outcomes.

# Third experiment: template-only grouping

Exact text and cosine similarity >= 0.90 define connected groups; event links are ignored. Same 2,625 tickets and five-outer/three-inner grouped evaluation. No threshold search was performed.

| Metric | Template-only |
|---|---:|
| Groups | 195 |
| Largest group | 126 tickets |
| Exact pair accuracy | 93.37% |
| Pair macro F1 | 87.69% |
| Category accuracy | 96.84% |
| Subcategory accuracy | 93.37% |
| Correct pairs | 2,451 / 2,625 |
| Incorrect pairs | 174 / 2,625 |

Test fold sizes are 537, 522, 522, 522, 522; 14 held-out tickets have pairs unseen in their corresponding training fold. These remain in evaluation. This has more balanced folds than the combined similarity/event experiment, but permits related incidents across folds by design. See event_cross_fold_audit.csv. Template families are semantic proxies, not verified generator IDs; chaining still applies. The synthetic corpus and repeated exploratory evaluations do not establish real-world accuracy.
