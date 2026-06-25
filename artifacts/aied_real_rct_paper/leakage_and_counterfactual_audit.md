# Leakage And Counterfactual Audit

## Audit Summary

| dataset | dataset_label | dataset_short | n_features | n_excluded_columns | n_post_treatment_dropped | uses_true_ate_in_training | uses_arm_means_in_training | uses_condition_as_feature |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assistments_rct88 | ASSISTments expanded RCT release | RCT88/89 | 30 | 12 | 2 | False | False | False |
| assistments_las2016 | LAS2016 22 randomized experiments | LAS2016 | 22 | 12 | 5 | False | False | False |
| assistments_abtest_study2 | ASSISTments OSF Study2 A/B tests | Study2 | 23 | 12 | 15 | False | False | False |

## Counterfactual Generation Rule

All methods are evaluated through the same do(T=0/1) rollout interface. The models do not receive true ATE labels, randomized arm means, or treatment condition as covariates.

## Real-RCT Metric Constraint

PEHE is not reported because real randomized logs reveal one realized outcome per student-task instance, not both individual potential outcomes.
