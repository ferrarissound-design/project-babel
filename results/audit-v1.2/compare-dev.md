| 条件 | 修正前 → 修正後（差のある指標のみ。▼は悪化） |
|---|---|
| levelA | 全指標で同一 |
| levelA:baseline_freq | 全指標で同一 |
| levelA:baseline_ngram | 全指標で同一 |
| levelA:docs=33 | 全指標で同一 |
| levelA:docs=65 | 全指標で同一 |
| levelA:heldout=late_era | 全指標で同一 |
| levelA:lacuna=0.0 | so_order_accuracy_when_answered 0.976→0.975<br>so_order_answered 0.683→0.667 ▼悪化 |
| levelA:lacuna=0.08 | so_order_accuracy_when_answered 0.944→0.943<br>so_order_answered 0.600→0.583 ▼悪化 |
| levelA:no_abstain | 全指標で同一 |
| levelA:no_em | 全指標で同一 |
| levelA:no_morph | 全指標で同一 |
| levelA:no_numeral_solver | 全指標で同一 |
| levelA:no_pos_constraint | 全指標で同一 |
| levelA:no_totals | 全指標で同一 |
| levelA:no_variants | 全指標で同一 |
| levelA:parallel=0.01 | 全指標で同一 |
| levelA:parallel=0.025 | 全指標で同一 |
| levelA:parallel=0.1 | 全指標で同一 |
| levelA:parallel=0.15 | 全指標で同一 |
| levelA:parallel=0.15:same_concept=0.4 | adj_noun_answered 0.267→0.283 |
| levelA:variation=0.0 | so_order_accuracy_when_answered 0.932→0.930<br>so_order_answered 0.733→0.717 ▼悪化 |
| levelA:variation=0.9 | 全指標で同一 |
| levelB | 全指標で同一 |
| levelB:no_metadata | 全指標で同一 |
| levelC | 全指標で同一 |
| levelC:baseline_freq | 全指標で同一 |
| levelC:baseline_ngram | 全指標で同一 |
| levelC:docs=33 | 全指標で同一 |
| levelC:docs=65 | 全指標で同一 |
| levelC:no_numeral_solver | 全指標で同一 |
| levelC:no_totals | 全指標で同一 |
| levelC:no_variants | 全指標で同一 |
| no_word_boundaries | 全指標で同一 |

差のあった指標の数: 7。悪化（0.005 超）: 3 件
- ▼ levelA:lacuna=0.0 / so_order_answered: 0.6833 → 0.6667
- ▼ levelA:lacuna=0.08 / so_order_answered: 0.6000 → 0.5833
- ▼ levelA:variation=0.0 / so_order_answered: 0.7333 → 0.7167
