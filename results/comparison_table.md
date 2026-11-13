# Reproduction results

Values are the mean over trials of each trial's scenario mean; ± is the standard deviation across trials (shown when there are at least two trials). Model and settings: see harness_mu/config/my_experiment.yaml, max-turns = 10.

| Attack | System | Trials | Privacy | Utility | Overall | Raw privacy | CC Δ privacy | CC intercepts / scenario | Paper P / U / O |
|---|---|---|---|---|---|---|---|---|---|
| none | Baseline (no harness) | 1 | 1.000 | 0.650 | 0.685 | n/a | n/a | n/a | 0.815 / 0.607 / 0.456 |
| none | Harness-MU (full) | 1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | 0.333 | 1.000 / 0.989 / 0.994 |
| fake_authorized | Baseline (no harness) | 1 | 1.000 | 0.590 | 0.631 | n/a | n/a | n/a | 0.926 / 0.506 / 0.475 |
| fake_authorized | Harness-MU (full) | 1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | 0.000 | 1.000 / 0.972 / 0.981 |
| fake_authorized | Harness-MU (no ComplianceChecker) | 1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | 0.000 | n/a |

## Per-trial scores

| Attack | System | Trial | Scenarios | Privacy | Utility | Overall | Raw privacy | Never scored |
|---|---|---|---|---|---|---|---|---|
| none | Baseline (no harness) | trial_1 | 18/18 | 1.000 | 0.650 | 0.685 | n/a | 0 |
| none | Harness-MU (full) | trial_1 | 18/18 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| fake_authorized | Baseline (no harness) | trial_1 | 18/18 | 1.000 | 0.590 | 0.631 | n/a | 0 |
| fake_authorized | Harness-MU (full) | trial_1 | 18/18 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| fake_authorized | Harness-MU (no ComplianceChecker) | trial_1 | 18/18 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |

"Never scored" counts baseline scenarios in which no agent reply was evaluated (zero entries in per_turn_metrics). The baseline records these with its default scores: privacy 1.0, utility 0.0.
