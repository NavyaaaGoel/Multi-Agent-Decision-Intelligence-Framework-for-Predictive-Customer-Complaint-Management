# Results (auto-generated)

> Synthetic data, chronological split (train 70% / val 15% / test 15%). See README for what this does and does not prove.

Test set: 1350 complaints, breach rate 18.7%.

## 1. SLA breach prediction

| Model | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Ensemble (used in pipeline)** | 0.9123 | 0.7175 | 0.0895 | 0.5687 | 0.8182 | 0.671 |
| Gradient boosting (calibrated) | 0.9003 | 0.6731 | 0.0955 | 0.7214 | 0.3992 | 0.514 |
| Logistic regression | 0.9156 | 0.7308 | 0.0873 | 0.7022 | 0.4941 | 0.58 |
| Distributional (log-normal time model) | 0.9036 | 0.6946 | 0.0928 | 0.6667 | 0.4822 | 0.5596 |
| Static rule (High/Critical & escalated) | 0.8387 | - | - | 0.4088 | 0.5138 | 0.4553 |

Flagging the top 20% riskiest complaints captures **66%** of all breaches (lift 3.3202x).

## 2. Resolution-time forecast

| Model | MAE (h) | R2 (log) |
|---|---|---|
| Gradient boosting | 39.26 | 0.9137 |
| Route x severity median baseline | 46.17 | 0.8832 |

P90 interval coverage: **89.8%** (nominal 90%).

## 3. Customer-escalation risk

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| gradient_boosting | 0.6853 | 0.3352 | 0.1201 |
| logistic_regression | 0.7122 | 0.3662 | 0.1161 |

## 4. Intake classification

| Model | Product acc. | Issue acc. |
|---|---|---|
| TF-IDF + LogReg (new) | 0.9763 | 0.9578 |
| Keyword matcher (hackathon) | 0.46 | 0.44 |

Share routed to human review (confidence < 0.4): 3.7%.

## 5. Top drivers of breach risk (permutation importance)

| Feature | AUC drop |
|---|---|
| sla_hours | 0.0894 |
| issue_key | 0.0597 |
| route | 0.0404 |
| team_load | 0.0335 |
| fast_tracked | 0.0242 |
| team | 0.0057 |
| repeat_count | 0.0053 |
| log_amount | 0.0049 |
| severity_score | 0.004 |
| tier | 0.0026 |

## 6. Policy simulation (held-out period, common random numbers)

| Policy | Breach rate | Crit/High breach rate | Mean hrs | Fast-tracks | Reroutes | Intervention cost | Net loss |
|---|---|---|---|---|---|---|---|
| no_intervention | 22.1% | 48.3% | 111.26 | 0 | 0 | 0.0 | 1618.2 |
| static_rule_fast_track_high_critical | 15.9% | 24.3% | 104.45 | 318 | 0 | 192.5 | 1186.9 |
| predictive_multi_agent | 12.5% | 24.9% | 97.36 | 431 | 75 | 276.2 | 1113.6 |

**predictive_vs_no_intervention**: net-loss reduction 31.2% (95% CI 24.2% to 37.4%); breaches avoided 129 (95% CI 109 to 150).

**predictive_vs_static_rule**: net-loss reduction 6.2% (95% CI 1.7% to 10.9%); breaches avoided 46 (95% CI 33 to 61).

Pipeline: 1350 complaints, verification pass rate 100.0%, human-review share 3.7%, mean latency 675.5 ms, end-to-end intake issue accuracy 95.8%.
Interventions chosen: {'standard': 895, 'fast_track': 378, 'fast_track+reroute': 67, 'reroute': 10}. Escalation levels: {'None': 986, 'Branch': 274, 'Regional Office': 77, 'MD/CEO Desk': 13}.
