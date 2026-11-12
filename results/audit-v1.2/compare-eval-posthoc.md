| 条件 | 修正前 → 修正後（差のある指標のみ。▼は悪化） |
|---|---|
| levelA | pos_accuracy 0.862→0.862 |
| levelA:baseline_freq | pos_accuracy 0.496→0.496 |
| levelA:baseline_ngram | pos_accuracy 0.252→0.251 |
| levelA:docs=33 | 全指標で同一 |
| levelA:docs=65 | 全指標で同一 |
| levelA:heldout=late_era | 全指標で同一 |
| levelA:lacuna=0.0 | pos_accuracy 0.876→0.876 |
| levelA:lacuna=0.08 | 全指標で同一 |
| levelA:no_abstain | pos_accuracy 0.862→0.862 |
| levelA:no_em | 全指標で同一 |
| levelA:no_morph | 全指標で同一 |
| levelA:no_numeral_solver | pos_accuracy 0.760→0.760 |
| levelA:no_pos_constraint | so_order_answered 0.483→0.467 ▼悪化 |
| levelA:no_totals | 全指標で同一 |
| levelA:no_variants | pos_accuracy 0.709→0.709<br>so_order_accuracy_when_answered 0.903→0.900<br>so_order_answered 0.517→0.500 ▼悪化 |
| levelA:parallel=0.01 | 全指標で同一 |
| levelA:parallel=0.025 | pos_accuracy 0.848→0.848 |
| levelA:parallel=0.1 | 全指標で同一 |
| levelA:parallel=0.15 | 全指標で同一 |
| levelA:parallel=0.15:same_concept=0.4 | adj_noun_accuracy_when_answered 0.963→0.962<br>adj_noun_answered 0.450→0.433 ▼悪化 |
| levelA:variation=0.0 | pos_accuracy 0.889→0.889 |
| levelA:variation=0.9 | pos_accuracy 0.862→0.862 |
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

差のあった指標の数: 15。悪化（0.005 超）: 3 件
- ▼ levelA:no_pos_constraint / so_order_answered: 0.4833 → 0.4667
- ▼ levelA:no_variants / so_order_answered: 0.5167 → 0.5000
- ▼ levelA:parallel=0.15:same_concept=0.4 / adj_noun_answered: 0.4500 → 0.4333
