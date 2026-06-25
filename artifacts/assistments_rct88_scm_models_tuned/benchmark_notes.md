# ASSISTments Real-RCT Six-Model Benchmark

This run uses ASSISTments RCT88/89 as the real-intervention benchmark.

## Models

- scm_causal
- rcgan
- vae
- diffusion
- crn
- timegan

## Target

The repository models are binary-outcome sequence models. For the processed ASSISTments RCT outcomes, the benchmark uses a binary target: completion uses assignment completion, and other standardized learning outcomes use `normalized_student_learning > 0`. The RCT ATE is computed on this same binary target.

## Data

- Dataset: ASSISTments RCT88/89.
- Analyzable binary RCT tasks: 86.
- Processed experiments represented: 44.
- Seeds: 42,43,44.
- Mean biased-sample retention: 0.525.

## Current Ranking

- Best ATE abs err: scm_causal (0.0369).
- Best policy value abs err: crn (0.0319).
- Best policy regret: scm_causal (0.0083).

## PEHE

PEHE is not reported for this pure real-RCT benchmark because individual-level potential outcomes are not both observed. The result table keeps `pehe` as missing and `pehe_supported=0`; PEHE should be reported on synthetic or semi-synthetic datasets with known individual treatment effects.
