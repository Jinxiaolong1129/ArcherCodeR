# Experiment Results Analysis

> Generated from `output/self-rl-jxl/`
> Ready = aime2024 + aime2025 + lcbv5_csv + lcbv6_csv all present for that step

---

## Overview

- Total experiments: 29
- **Fully ready** (all 4 steps 10/50/80/105, all 4 datasets): **24**
- Partially ready: 3
- Not fully ready total: 5 (partial 3 + zero-ready 2)

---

## Experiment Design

### Model


| Item            | Value                                       |
| --------------- | ------------------------------------------- |
| Base model      | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Parameter count | 1.5B                                        |


### Training Data


| File                                     | Domain                         | Notes                            |
| ---------------------------------------- | ------------------------------ | -------------------------------- |
| `data/train/archercoder-1.5b-train.json` | Code (competitive programming) | Shared across all 29 experiments |


### Evaluation Datasets


| Dataset          | Domain | Format                                | nsamples | Inference length | Notes            |
| ---------------- | ------ | ------------------------------------- | -------- | ---------------- | ---------------- |
| AIME 2024        | Math   | Parquet (126 files)                   | 32       | 8k output        | Competition math |
| AIME 2025        | Math   | Parquet (124 files)                   | 32       | 8k output        | Competition math |
| LiveCodeBench v5 | Code   | CSV — 103 files (8k) + 20 files (16k) | 8        | 8k or 16k        | pass@1/4/8       |
| LiveCodeBench v6 | Code   | CSV — 104 files (8k) + 20 files (16k) | 8        | 8k or 16k        | pass@1/4/8       |


> **Ready standard**: a checkpoint step is considered ready when all four of {aime2024, aime2025, lcbv5.csv, lcbv6.csv} are present for that step.

### Context Lengths


| Setting                              | Value            | Scope                |
| ------------------------------------ | ---------------- | -------------------- |
| Max prompt length                    | 2K (2048 tokens) | Training             |
| Max response length                  | 8K (8192 tokens) | Training             |
| Total context window                 | 10K tokens       | Training             |
| Inference response length (standard) | 8K               | Eval — AIME + LCB-8k |
| Inference response length (extended) | 16K              | Eval — LCB-16k only  |


> **Note**: Only 5 experiments (the 5 "baseline" runs, one per method) have full 16k eval coverage. All other ablation variants currently only have 8k eval results.

---

## Unified Training Hyperparameters

All 29 experiments share the following base settings:


| Hyperparameter                       | Value                                 |
| ------------------------------------ | ------------------------------------- |
| Optimizer LR                         | `1e-6`                                |
| Warmup style                         | `constant`                            |
| LR warmup steps                      | `10`                                  |
| Weight decay                         | `0.1`                                 |
| PPO epochs                           | `1`                                   |
| `clip_ratio_c`                       | `3.0` (except c10-e3 variant)         |
| `clip_ratio_low` / `clip_ratio_high` | `0.2 / 0.2`                           |
| Train batch size (prompts)           | `64`                                  |
| Mini-batch size                      | `32`                                  |
| Responses per prompt (`n`)           | `16` (except n8/n12 variants)         |
| Temperature (train rollout)          | `1.0` (except temp08/temp12 variants) |
| Val temperature                      | `0.8`                                 |
| Val n                                | `4`                                   |
| Total training epochs                | `2` (≈ 105 gradient steps)            |
| Save/test frequency                  | every 10 steps                        |
| Tensor parallel size                 | `2`                                   |
| GPUs per node                        | `8`                                   |
| Nodes                                | `1`                                   |
| KL loss                              | disabled (except kl0005 variants)     |


---

## Experiment Taxonomy

### 5 Core Methods (Baseline, No Ablation)


| Experiment                            | Algorithm          | KL  | n   | Temperature | clipc | ppoepochs | 8k eval | 16k eval |
| ------------------------------------- | ------------------ | --- | --- | ----------- | ----- | --------- | ------- | -------- |
| `Unified-Pure-GRPO-...-no-kl`         | Pure GRPO          | ✗   | 16  | 1.0         | 3.0   | 1         | ✓       | ✓        |
| `Unified-Intuitor-...-no-kl`          | Intuitor           | ✗   | 16  | 1.0         | 3.0   | 1         | ✓       | ✓        |
| `Unified-TrajectoryEntropy-...-no-kl` | Trajectory Entropy | ✗   | 16  | 1.0         | 3.0   | 1         | ✓       | ✓        |
| `Unified-TokenEntropy-...-no-kl`      | Token Entropy      | ✗   | 16  | 1.0         | 3.0   | 1         | ✓       | ✓        |
| `Unified-ProbDisparity-...-no-kl`     | Prob Disparity     | ✗   | 16  | 1.0         | 3.0   | 1         | ✓       | ✓        |


### Temperature Sweep (×5 methods × 2 temps = 10 experiments)


| Experiment suffix | Temperature | Methods                                                             |
| ----------------- | ----------- | ------------------------------------------------------------------- |
| `-temp08`         | 0.8         | Pure-GRPO, Intuitor, TrajectoryEntropy, TokenEntropy, ProbDisparity |
| `-temp12`         | 1.2         | Pure-GRPO, Intuitor, TrajectoryEntropy, TokenEntropy, ProbDisparity |


> 8k eval only; no 16k eval for temp variants.

### Response-Count Sweep (×5 methods × 2 values = 10 experiments)


| Experiment suffix | n (responses/prompt) | Methods                                                             |
| ----------------- | -------------------- | ------------------------------------------------------------------- |
| `-n8`             | 8                    | Pure-GRPO, Intuitor, TrajectoryEntropy, TokenEntropy, ProbDisparity |
| `-n12`            | 12                   | Pure-GRPO, Intuitor, TrajectoryEntropy, TokenEntropy, ProbDisparity |


> 8k eval only; no 16k eval for n-sweep variants.

### KL Loss Sweep (×5 methods × 1 coef = 5 experiments)


| Experiment suffix | KL enabled | KL coef | Methods                                                              |
| ----------------- | ---------- | ------- | -------------------------------------------------------------------- |
| `-kl0005`         | ✓          | 0.005   | Pure-GRPO¹, Intuitor, TrajectoryEntropy, TokenEntropy, ProbDisparity |


> ¹ Pure-GRPO-kl0005 is not listed among the 29 experiments — KL sweep was only completed for the 4 self-RL methods.
> 8k eval only; no 16k eval for KL variants.

### Special Pure-GRPO Variants (2 experiments)


| Experiment                      | clipc | ppoepochs | lossaggmode    | Notes                               |
| ------------------------------- | ----- | --------- | -------------- | ----------------------------------- |
| `...-clip_ratio_c10-ppo_epoch3` | 10.0  | 3         | token-mean     | Mimics original Pure-GRPO defaults  |
| `...-token-separate-c3-e3`      | 3.0   | 3         | token-separate | Separate clipping for token entropy |


---

## Full Experiment List with Configuration


| #   | Experiment                                        | Method         | KL      | n   | Temp | clipc | Fully Ready | Ready Steps  |
| --- | ------------------------------------------------- | -------------- | ------- | --- | ---- | ----- | ----------- | ------------ |
| 1   | `Unified-Pure-GRPO-...-no-kl`                     | Pure GRPO      | ✗       | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 2   | `Unified-Pure-GRPO-...-no-kl-n8`                  | Pure GRPO      | ✗       | 8   | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 3   | `Unified-Pure-GRPO-...-no-kl-n12`                 | Pure GRPO      | ✗       | 12  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 4   | `Unified-Pure-GRPO-...-clip_ratio_c10-ppo_epoch3` | Pure GRPO      | ✗       | 16  | 1.0  | 10.0  | ✗           | (none fully) |
| 5   | `Unified-Pure-GRPO-...-token-separate-c3-e3`      | Pure GRPO      | ✗       | 16  | 1.0  | 3.0   | —           | (not in 28)  |
| 6   | `Unified-Intuitor-...-no-kl`                      | Intuitor       | ✗       | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 7   | `Unified-Intuitor-...-no-kl-temp08`               | Intuitor       | ✗       | 16  | 0.8  | 3.0   | ✓           | 10,50,80,105 |
| 8   | `Unified-Intuitor-...-no-kl-temp12`               | Intuitor       | ✗       | 16  | 1.2  | 3.0   | ✓           | 10,50,80,105 |
| 9   | `Unified-Intuitor-...-no-kl-n8`                   | Intuitor       | ✗       | 8   | 1.0  | 3.0   | Partial     | 10,50,80     |
| 10  | `Unified-Intuitor-...-no-kl-n12`                  | Intuitor       | ✗       | 12  | 1.0  | 3.0   | Partial     | 10,50        |
| 11  | `Unified-Intuitor-...-no-kl-kl0005`               | Intuitor       | ✓ 0.005 | 16  | 1.0  | 3.0   | Partial     | 10,50,80     |
| 12  | `Unified-TrajectoryEntropy-...-no-kl`             | Traj. Entropy  | ✗       | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 13  | `Unified-TrajectoryEntropy-...-no-kl-temp08`      | Traj. Entropy  | ✗       | 16  | 0.8  | 3.0   | ✓           | 10,50,80,105 |
| 14  | `Unified-TrajectoryEntropy-...-no-kl-temp12`      | Traj. Entropy  | ✗       | 16  | 1.2  | 3.0   | ✓           | 10,50,80,105 |
| 15  | `Unified-TrajectoryEntropy-...-no-kl-n8`          | Traj. Entropy  | ✗       | 8   | 1.0  | 3.0   | Partial     | 10,50,80     |
| 16  | `Unified-TrajectoryEntropy-...-no-kl-n12`         | Traj. Entropy  | ✗       | 12  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 17  | `Unified-TrajectoryEntropy-...-no-kl-kl0005`      | Traj. Entropy  | ✓ 0.005 | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 18  | `Unified-TokenEntropy-...-no-kl`                  | Token Entropy  | ✗       | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 19  | `Unified-TokenEntropy-...-no-kl-temp08`           | Token Entropy  | ✗       | 16  | 0.8  | 3.0   | ✓           | 10,50,80,105 |
| 20  | `Unified-TokenEntropy-...-no-kl-temp12`           | Token Entropy  | ✗       | 16  | 1.2  | 3.0   | ✓           | 10,50,80,105 |
| 21  | `Unified-TokenEntropy-...-no-kl-n8`               | Token Entropy  | ✗       | 8   | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 22  | `Unified-TokenEntropy-...-no-kl-n12`              | Token Entropy  | ✗       | 12  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 23  | `Unified-TokenEntropy-...-no-kl-kl0005`           | Token Entropy  | ✓ 0.005 | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 24  | `Unified-ProbDisparity-...-no-kl`                 | Prob Disparity | ✗       | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 25  | `Unified-ProbDisparity-...-no-kl-temp08`          | Prob Disparity | ✗       | 16  | 0.8  | 3.0   | ✓           | 10,50,80,105 |
| 26  | `Unified-ProbDisparity-...-no-kl-temp12`          | Prob Disparity | ✗       | 16  | 1.2  | 3.0   | ✓           | 10,50,80,105 |
| 27  | `Unified-ProbDisparity-...-no-kl-n8`              | Prob Disparity | ✗       | 8   | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| 28  | `Unified-ProbDisparity-...-no-kl-n12`             | Prob Disparity | ✗       | 12  | 1.0  | 3.0   | ✓           | 10,50,80,105 |
| —   | `Unified-ProbDisparity-...-no-kl-kl0005`          | Prob Disparity | ✓ 0.005 | 16  | 1.0  | 3.0   | ✓           | 10,50,80,105 |


> Note: `Unified-ProbDisparity-...-kl0005` is counted in the original 23 fully-ready experiments. The table above uses `...` as shorthand for `Qwen2.5-1.5B-2k-8k-batch64` (self-RL methods) or `Qwen2.5-1.5B-2K-8K-16resp` (Pure-GRPO).

---

## Summary

---

## Fully Ready Experiments (24)

All 4 target steps (10, 50, 80, 105), all 4 datasets (aime2024, aime2025, lcbv5, lcbv6) confirmed.


| Experiment                                                          | Available steps | aime24 | aime25 | lcbv5 8k | lcbv5 16k | lcbv6 8k | lcbv6 16k |
| ------------------------------------------------------------------- | --------------- | ------ | ------ | -------- | --------- | -------- | --------- |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`                 | 10,50,80,105    | ✓      | ✓      | ✓        | ✓         | ✓        | ✓         |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`          | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`          | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`            | 10,50,80,105    | ✓      | ✓      | ✓        | ✓         | ✓        | ✓         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`     | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`        | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`         | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`     | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`     | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`                 | 10,50,80,105    | ✓      | ✓      | ✓        | ✓         | ✓        | ✓         |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`             | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`              | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`             | 10,50,80,105    | ✓      | ✓      | ✓        | ✓         | ✓        | ✓         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`      | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`         | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`          | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`      | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`      | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`        | 10,50,80,105    | ✓      | ✓      | ✓        | ✓         | ✓        | ✓         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005` | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`     | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`    | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08` | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12` | 10,50,80,105    | ✓      | ✓      | ✓        | —         | ✓        | —         |


---

## Not Fully Ready Experiments (5)


| Experiment                                                                    | Ready steps | Missing steps | Notes |
| ----------------------------------------------------------------------------- | ----------- | ------------- | ----- |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`                    | 10,50,80    | 105           | step105 4项都缺 |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`                       | 10,50       | 80,105        | step80/105 4项都缺 |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`                        | 10,50,80    | 105           | step105 4项都缺 |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3` | —           | 10,50,80,105  | 每步仅缺 lcbv6 |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-token-separate-c3-e3`      | —           | 10,50,80,105  | step10缺v5+v6; 50/80缺v5; 105缺v6 |


---

## LCB Metrics Summary (pass@1)

### 8k Inference (output_livecodebench_*_8k_n8)


| Experiment                                                          | step | lcbv5 pass@1 | lcbv5 pass@8 | lcbv6 pass@1 | lcbv6 pass@8 | avg_v5v6 pass@1 |
| ------------------------------------------------------------------- | ---- | ------------ | ------------ | ------------ | ------------ | --------------- |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`          | 50   | 0.1635       | 0.2832       | 0.1174       | 0.1985       | 0.1404          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`     | 50   | 0.1613       | 0.2903       | 0.1193       | 0.2290       | 0.1403          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`      | 50   | 0.1689       | 0.2867       | 0.1088       | 0.2061       | 0.1388          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`        | 50   | 0.1631       | 0.2939       | 0.1145       | 0.2366       | 0.1388          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`             | 50   | 0.1635       | 0.2867       | 0.1135       | 0.2137       | 0.1385          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`         | 50   | 0.1662       | 0.3047       | 0.1078       | 0.2137       | 0.1370          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005` | 50   | 0.1595       | 0.2616       | 0.1126       | 0.1908       | 0.1360          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`      | 50   | 0.1631       | 0.3011       | 0.1088       | 0.1832       | 0.1359          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`                 | 80   | 0.1631       | 0.3011       | 0.1059       | 0.1908       | 0.1345          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`      | 80   | 0.1546       | 0.2760       | 0.1135       | 0.2214       | 0.1341          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`         | 10   | 0.1626       | 0.2832       | 0.1050       | 0.1832       | 0.1338          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`      | 50   | 0.1582       | 0.2903       | 0.1088       | 0.2061       | 0.1335          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`             | 10   | 0.1577       | 0.2652       | 0.1078       | 0.2061       | 0.1328          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`    | 10   | 0.1568       | 0.2652       | 0.1078       | 0.1603       | 0.1323          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`              | 80   | 0.1613       | 0.2832       | 0.1021       | 0.1908       | 0.1317          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`              | 50   | 0.1532       | 0.2581       | 0.1088       | 0.2061       | 0.1310          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`          | 80   | 0.1541       | 0.2724       | 0.1069       | 0.2061       | 0.1305          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`     | 80   | 0.1564       | 0.2832       | 0.1040       | 0.1908       | 0.1302          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`     | 10   | 0.1510       | 0.2652       | 0.1078       | 0.1908       | 0.1294          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`      | 80   | 0.1465       | 0.2652       | 0.1116       | 0.1908       | 0.1291          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005` | 10   | 0.1541       | 0.2581       | 0.1040       | 0.1756       | 0.1291          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`      | 10   | 0.1510       | 0.2796       | 0.1040       | 0.1832       | 0.1275          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`      | 105  | 0.1447       | 0.2509       | 0.1097       | 0.1679       | 0.1272          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`      | 80   | 0.1411       | 0.2473       | 0.1126       | 0.2061       | 0.1269          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`      | 10   | 0.1470       | 0.2401       | 0.1050       | 0.1756       | 0.1260          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`        | 105  | 0.1371       | 0.2509       | 0.1145       | 0.2061       | 0.1258          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`          | 105  | 0.1492       | 0.2760       | 0.1021       | 0.1908       | 0.1256          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`          | 10   | 0.1510       | 0.2616       | 0.1002       | 0.1603       | 0.1256          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`                 | 10   | 0.1501       | 0.2760       | 0.1002       | 0.1679       | 0.1251          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`      | 105  | 0.1384       | 0.2437       | 0.1097       | 0.1985       | 0.1241          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`         | 80   | 0.1465       | 0.2867       | 0.1011       | 0.1832       | 0.1238          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005` | 80   | 0.1425       | 0.2545       | 0.1021       | 0.1908       | 0.1223          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12` | 80   | 0.1393       | 0.2652       | 0.1050       | 0.1908       | 0.1221          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`    | 105  | 0.1340       | 0.2581       | 0.1097       | 0.1832       | 0.1218          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`      | 10   | 0.1523       | 0.2832       | 0.0906       | 0.1603       | 0.1215          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`        | 10   | 0.1474       | 0.2832       | 0.0954       | 0.1756       | 0.1214          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`             | 105  | 0.1425       | 0.2545       | 0.1002       | 0.1832       | 0.1213          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`             | 80   | 0.1322       | 0.2437       | 0.1078       | 0.1908       | 0.1200          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`         | 105  | 0.1335       | 0.2437       | 0.1011       | 0.1832       | 0.1173          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005` | 105  | 0.1353       | 0.2509       | 0.0973       | 0.1679       | 0.1163          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12` | 105  | 0.1228       | 0.2330       | 0.0992       | 0.1756       | 0.1110          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`      | 105  | 0.1228       | 0.2186       | 0.0964       | 0.1527       | 0.1096          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`     | 50   | 0.0999       | 0.1864       | 0.1135       | 0.1908       | 0.1067          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`             | 105  | 0.0932       | 0.1685       | 0.1193       | 0.2061       | 0.1062          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`             | 50   | 0.0927       | 0.1685       | 0.1174       | 0.2061       | 0.1051          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`         | 50   | 0.0901       | 0.1685       | 0.1193       | 0.2137       | 0.1047          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`              | 105  | 0.1008       | 0.1900       | 0.1059       | 0.1756       | 0.1034          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`     | 50   | 0.0905       | 0.1720       | 0.1145       | 0.1832       | 0.1025          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`            | 50   | 0.0914       | 0.1864       | 0.1135       | 0.2137       | 0.1025          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`             | 50   | 0.0923       | 0.1577       | 0.1116       | 0.1908       | 0.1020          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`          | 50   | 0.0941       | 0.1649       | 0.1097       | 0.2214       | 0.1019          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`          | 50   | 0.0892       | 0.1649       | 0.1145       | 0.2061       | 0.1018          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`                 | 105  | 0.0936       | 0.1720       | 0.1097       | 0.1908       | 0.1017          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`            | 80   | 0.0829       | 0.1470       | 0.1202       | 0.1908       | 0.1016          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08` | 50   | 0.0932       | 0.1900       | 0.1097       | 0.1985       | 0.1015          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`             | 80   | 0.0936       | 0.1685       | 0.1088       | 0.1908       | 0.1012          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`    | 50   | 0.0905       | 0.1685       | 0.1116       | 0.1985       | 0.1011          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`        | 50   | 0.0923       | 0.1685       | 0.1088       | 0.1985       | 0.1005          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12` | 50   | 0.0887       | 0.1613       | 0.1107       | 0.1908       | 0.0997          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`                 | 50   | 0.0887       | 0.1685       | 0.1107       | 0.1985       | 0.0997          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`         | 80   | 0.0914       | 0.1649       | 0.1078       | 0.2061       | 0.0996          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08` | 80   | 0.0883       | 0.1720       | 0.1107       | 0.2061       | 0.0995          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`              | 80   | 0.0833       | 0.1505       | 0.1155       | 0.2443       | 0.0994          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`          | 50   | 0.0932       | 0.1756       | 0.1050       | 0.1908       | 0.0991          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`     | 50   | 0.0851       | 0.1613       | 0.1116       | 0.2061       | 0.0984          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`             | 10   | 0.0927       | 0.1649       | 0.1040       | 0.1679       | 0.0984          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`     | 10   | 0.0878       | 0.1649       | 0.1088       | 0.1832       | 0.0983          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`              | 50   | 0.0887       | 0.1756       | 0.1078       | 0.1756       | 0.0983          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`                 | 50   | 0.0936       | 0.1613       | 0.1002       | 0.1679       | 0.0969          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`     | 80   | 0.0847       | 0.1505       | 0.1088       | 0.2214       | 0.0967          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`    | 80   | 0.0842       | 0.1613       | 0.1088       | 0.2061       | 0.0965          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`              | 10   | 0.0878       | 0.1649       | 0.1050       | 0.1756       | 0.0964          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08` | 105  | 0.0824       | 0.1505       | 0.1097       | 0.1908       | 0.0961          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`              | 10   | 0.0865       | 0.1649       | 0.1050       | 0.1832       | 0.0957          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`            | 10   | 0.0883       | 0.1577       | 0.1031       | 0.1832       | 0.0957          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`        | 80   | 0.0892       | 0.1792       | 0.1021       | 0.1908       | 0.0956          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`         | 105  | 0.0824       | 0.1541       | 0.1078       | 0.1756       | 0.0951          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12` | 10   | 0.0860       | 0.1577       | 0.1040       | 0.1756       | 0.0950          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`          | 10   | 0.0878       | 0.1470       | 0.1021       | 0.1756       | 0.0950          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`         | 10   | 0.0856       | 0.1505       | 0.1040       | 0.1756       | 0.0948          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`          | 10   | 0.0851       | 0.1470       | 0.1040       | 0.1908       | 0.0946          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`     | 10   | 0.0856       | 0.1577       | 0.1021       | 0.1756       | 0.0938          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`                 | 10   | 0.0878       | 0.1505       | 0.0992       | 0.1603       | 0.0935          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`        | 105  | 0.0833       | 0.1577       | 0.1031       | 0.1756       | 0.0932          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`            | 105  | 0.0806       | 0.1434       | 0.1050       | 0.1832       | 0.0928          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`     | 80   | 0.0887       | 0.1756       | 0.0964       | 0.1832       | 0.0925          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`             | 10   | 0.0896       | 0.1649       | 0.0954       | 0.1450       | 0.0925          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`        | 10   | 0.0905       | 0.1613       | 0.0945       | 0.1527       | 0.0925          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08` | 10   | 0.0847       | 0.1470       | 0.1002       | 0.1679       | 0.0924          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`     | 105  | 0.0842       | 0.1577       | 0.1002       | 0.1679       | 0.0922          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`        | 80   | 0.0744       | 0.1398       | 0.1097       | 0.2061       | 0.0921          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`     | 10   | 0.0820       | 0.1505       | 0.1002       | 0.1832       | 0.0911          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`          | 80   | 0.0829       | 0.1505       | 0.0983       | 0.1985       | 0.0906          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`          | 10   | 0.0869       | 0.1613       | 0.0935       | 0.1527       | 0.0902          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`          | 80   | 0.0735       | 0.1290       | 0.1050       | 0.1756       | 0.0892          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`     | 80   | 0.0766       | 0.1434       | 0.1011       | 0.1908       | 0.0889          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`     | 105  | 0.0780       | 0.1505       | 0.0992       | 0.1832       | 0.0886          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`          | 80   | 0.0793       | 0.1541       | 0.0973       | 0.1679       | 0.0883          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`          | 105  | 0.0757       | 0.1470       | 0.0973       | 0.1756       | 0.0865          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`     | 105  | 0.0708       | 0.1219       | 0.1011       | 0.1527       | 0.0860          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`                 | 80   | 0.0748       | 0.1398       | 0.0954       | 0.1679       | 0.0851          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`          | 105  | 0.0659       | 0.1470       | 0.1002       | 0.1832       | 0.0830          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`                 | 105  | 0.0703       | 0.1362       | 0.0906       | 0.1603       | 0.0805          |


### 16k Inference (output_16k/livecodebench_*_n8)


| Experiment                                                   | step | lcbv5 pass@1 | lcbv5 pass@8 | lcbv6 pass@1 | lcbv6 pass@8 | avg_v5v6 pass@1 |
| ------------------------------------------------------------ | ---- | ------------ | ------------ | ------------ | ------------ | --------------- |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`          | 80   | 0.1788       | 0.3082       | 0.1164       | 0.2290       | 0.1476          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`      | 10   | 0.1653       | 0.2939       | 0.1202       | 0.1985       | 0.1428          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl` | 50   | 0.1658       | 0.3333       | 0.1174       | 0.1985       | 0.1416          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`      | 50   | 0.1640       | 0.2832       | 0.1107       | 0.1985       | 0.1373          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`          | 10   | 0.1568       | 0.2867       | 0.1174       | 0.2137       | 0.1371          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl` | 80   | 0.1465       | 0.2688       | 0.1069       | 0.1832       | 0.1267          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`      | 105  | 0.1326       | 0.2366       | 0.1097       | 0.2137       | 0.1212          |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`      | 80   | 0.1402       | 0.2652       | 0.0992       | 0.1832       | 0.1197          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`          | 105  | 0.1053       | 0.1828       | 0.1326       | 0.2443       | 0.1190          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl` | 105  | 0.1295       | 0.2294       | 0.1069       | 0.1985       | 0.1182          |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`          | 50   | 0.1008       | 0.2007       | 0.1279       | 0.2366       | 0.1143          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`     | 10   | 0.1004       | 0.1720       | 0.1231       | 0.2061       | 0.1117          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`          | 10   | 0.1017       | 0.1792       | 0.1212       | 0.2290       | 0.1114          |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl` | 10   | 0.0959       | 0.1900       | 0.1174       | 0.2137       | 0.1066          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`     | 50   | 0.1017       | 0.1900       | 0.1059       | 0.1908       | 0.1038          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`          | 50   | 0.0918       | 0.1720       | 0.1097       | 0.2214       | 0.1008          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`     | 80   | 0.0878       | 0.1720       | 0.1050       | 0.1908       | 0.0964          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`          | 80   | 0.0797       | 0.1434       | 0.1031       | 0.1908       | 0.0914          |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`     | 105  | 0.0789       | 0.1470       | 0.1011       | 0.1603       | 0.0900          |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`          | 105  | 0.0672       | 0.1362       | 0.1040       | 0.1603       | 0.0856          |


---

## Coverage by Dataset

| Dataset | Type | Parquet / CSV files | Notes |
|---|---|---|---|
| aime2024 | math | 126 parquets | n_samples=32, 8k output |
| aime2025 | math | 124 parquets | n_samples=32, 8k output |
| livecodebench_v5 | code | 103 CSVs (8k) + 20 CSVs (16k) | n_samples=8, pass@1/4/8 |
| livecodebench_v6 | code | 104 CSVs (8k) + 20 CSVs (16k) | n_samples=8, pass@1/4/8 |

---

## Full Eval Coverage Matrix (8k and 16k)

> Columns: `a24`/`a25` = AIME 2024/2025 (8k inference); `v5`/`v6` = LCBv5/v6 (8k); `a24L`/`a25L`/`v5L`/`v6L` = same datasets at 16k inference.
> Eval results live at: `global_step_{N}/actor/hf_model/output_{dataset}_8k_*/` and `output_16k/*/`

| Experiment | step | a24 | a25 | v5 | v6 | a24L | a25L | v5L | v6L |
|---|---|---|---|---|---|---|---|---|---|
| `Unified-Pure-GRPO-...-no-kl` | 10 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Pure-GRPO-...-no-kl` | 50 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Pure-GRPO-...-no-kl` | 80 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Pure-GRPO-...-no-kl` | 105 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Pure-GRPO-...-no-kl-n8` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n8` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n8` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n8` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-no-kl-n12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Pure-GRPO-...-clip_ratio_c10-ppo_epoch3` | 10 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-clip_ratio_c10-ppo_epoch3` | 50 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-clip_ratio_c10-ppo_epoch3` | 80 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-clip_ratio_c10-ppo_epoch3` | 105 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-token-separate-c3-e3` | 10 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-token-separate-c3-e3` | 50 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-token-separate-c3-e3` | 80 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Pure-GRPO-...-token-separate-c3-e3` | 105 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Intuitor-...-no-kl` | 10 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Intuitor-...-no-kl` | 50 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Intuitor-...-no-kl` | 80 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Intuitor-...-no-kl` | 105 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-Intuitor-...-temp08` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp08` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp08` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp08` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-temp12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n8` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n8` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n8` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n8` | 105 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Intuitor-...-n12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-n12` | 80 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Intuitor-...-n12` | 105 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-Intuitor-...-kl0005` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-kl0005` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-kl0005` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-Intuitor-...-kl0005` | 105 | **—** | **—** | **—** | **—** | — | — | — | — |
| `Unified-TrajectoryEntropy-...-no-kl` | 10 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TrajectoryEntropy-...-no-kl` | 50 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TrajectoryEntropy-...-no-kl` | 80 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TrajectoryEntropy-...-no-kl` | 105 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TrajectoryEntropy-...-temp08` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp08` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp08` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp08` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-temp12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n8` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n8` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n8` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n8` | 105 | Y | Y | **—** | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-n12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-kl0005` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-kl0005` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-kl0005` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TrajectoryEntropy-...-kl0005` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-no-kl` | 10 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TokenEntropy-...-no-kl` | 50 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TokenEntropy-...-no-kl` | 80 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TokenEntropy-...-no-kl` | 105 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-TokenEntropy-...-temp08` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp08` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp08` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp08` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-temp12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n8` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n8` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n8` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n8` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-n12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-kl0005` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-kl0005` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-kl0005` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-TokenEntropy-...-kl0005` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-no-kl` | 10 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-ProbDisparity-...-no-kl` | 50 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-ProbDisparity-...-no-kl` | 80 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-ProbDisparity-...-no-kl` | 105 | Y | Y | Y | Y | **Y** | **Y** | **Y** | **Y** |
| `Unified-ProbDisparity-...-temp08` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp08` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp08` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp08` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-temp12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n8` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n8` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n8` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n8` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n12` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n12` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n12` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-n12` | 105 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-kl0005` | 10 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-kl0005` | 50 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-kl0005` | 80 | Y | Y | Y | Y | — | — | — | — |
| `Unified-ProbDisparity-...-kl0005` | 105 | Y | Y | Y | Y | — | — | — | — |

> `...` is shorthand for the common name prefix of each method group.

---

## Missing Eval Summary

### 1. AIME Has 16k Results

AIME 2024 and AIME 2025 **do** have 16k inference results (`output_16k/aime2024_n32/` and `output_16k/aime2025_n32/`), but only for the **5 baseline experiments** (one per method, no ablation suffix).

**16k eval complete (all 4 datasets × 4 steps):**

| Experiment | 16k steps covered |
|---|---|
| `Unified-Pure-GRPO-...-no-kl` | 10, 50, 80, 105 |
| `Unified-Intuitor-...-no-kl` | 10, 50, 80, 105 |
| `Unified-TrajectoryEntropy-...-no-kl` | 10, 50, 80, 105 |
| `Unified-TokenEntropy-...-no-kl` | 10, 50, 80, 105 |
| `Unified-ProbDisparity-...-no-kl` | 10, 50, 80, 105 |

All other 24 experiments (ablation variants) have **no 16k eval at all**.

### 2. Missing 8k Evals (Incomplete Steps)

| Experiment | Missing steps | Missing datasets | Cause |
|---|---|---|---|
| `...-clip_ratio_c10-ppo_epoch3` | 10, 50, 80, 105 | all 4 | No eval run at all |
| `...-token-separate-c3-e3` | 10, 50, 80, 105 | all 4 | No eval run at all |
| `Unified-Intuitor-...-n8` | 105 | all 4 | Checkpoint not yet reached |
| `Unified-Intuitor-...-n12` | 80, 105 | all 4 | Checkpoint not yet reached |
| `Unified-Intuitor-...-kl0005` | 105 | all 4 | Checkpoint not yet reached |
| `Unified-TrajectoryEntropy-...-n8` | 105 | v5 only | LCBv5 result file missing |

### 3. 16k Eval Gap for Ablation Variants

All **24 ablation experiments** (temp/n/kl sweeps) have full 8k eval but zero 16k eval. If 16k results are needed for ablation comparisons, evals at all 4 target steps need to be run for each:

- All `-temp08` variants (×5 methods = 5 experiments)
- All `-temp12` variants (×5 methods = 5 experiments)
- All `-n8` variants (×5 methods = 5 experiments)
- All `-n12` variants (×5 methods = 5 experiments)
- All `-kl0005` variants (×4 self-RL methods = 4 experiments)


