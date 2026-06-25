# ASSISTments Real-RCT Six-Model Benchmark

This run uses ASSISTments LAS2016 22 RCTs as the real-intervention benchmark.

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

- Dataset: ASSISTments LAS2016 22 RCTs.
- Analyzable binary RCT tasks: 22.
- Processed experiments represented: 22.
- Seeds: 42,43,44.
- Mean biased-sample retention: 0.526.

## Current Ranking

- Best ATE abs err: vae (0.0201).
- Best policy value abs err: scm_causal (0.0189).
- Best policy regret: vae (0.0042).

## PEHE

PEHE is not reported for this pure real-RCT benchmark because individual-level potential outcomes are not both observed. The result table keeps `pehe` as missing and `pehe_supported=0`; PEHE should be reported on synthetic or semi-synthetic datasets with known individual treatment effects.
