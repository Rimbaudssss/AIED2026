# Real RCT Results Digest

## Scope

Only real randomized ASSISTments intervention datasets are included. Study1 from the OSF TACO release is the same 22-experiment LAS2016 data, so it is not double-counted as a third benchmark.

## Dataset Sources

- RCT88/89: 86 tasks, 76,216 student-task rows, source https://osf.io/m2jqe/
- LAS2016: 22 tasks, 14,947 student-task rows, source https://sites.google.com/site/las2016data/data/thison
- Study2: 11 tasks, 6,986 student-task rows, source https://osf.io/j6esa/

## Main Macro Results

- SCM-Causal: ATE 0.0337, value 0.0277, regret 0.0085, macro-rank 1.56
- CRN: ATE 0.0414, value 0.0304, regret 0.0145, macro-rank 3.11
- RCGAN: ATE 0.0404, value 0.0770, regret 0.0090, macro-rank 3.22
- VAE: ATE 0.0423, value 0.0615, regret 0.0132, macro-rank 3.28
- Diffusion: ATE 0.0451, value 0.1946, regret 0.0118, macro-rank 4.50
- TimeGAN: ATE 0.0472, value 0.0733, regret 0.0242, macro-rank 5.33

## SCM-Causal By Dataset

- RCT88/89: ATE 0.0369, policy value 0.0325, regret 0.0083, sign 0.725, Pearson 0.790
- LAS2016: ATE 0.0215, policy value 0.0189, regret 0.0056, sign 0.712, Pearson 0.355
- Study2: ATE 0.0427, policy value 0.0319, regret 0.0117, sign 0.727, Pearson 0.570

## Interpretation

The strongest overall method by macro rank is SCM-Causal. Baseline models can produce do(T=0/1) estimates because they implement the same generator rollout API, but the audit verifies that true ATEs, randomized arm means, and treatment condition are not used as training features.

PEHE is intentionally left undefined for these real RCTs because individual paired potential outcomes are not observed.
