# Evaluation Results Summary

> Generated from `output/self-rl-jxl/`

## Legend
- **✅** = parquet or csv exists  **❌** = missing
- AIME metrics: pass@32 (acc%)  | LCB metrics: pass@1 (%)
- 8k = max_response_length 8192  | 16k = max_response_length 16384

## Method: Intuitor

### `baseline`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.8 | ✅ 9.9 | ✅ |
| 50 | ✅ | ✅ | ✅ 8.9 | ✅ 11.1 | ✅ |
| 80 | ✅ | ✅ | ✅ 7.5 | ✅ 9.5 | ✅ |
| 105 | ✅ | ✅ | ✅ 7.0 | ✅ 9.1 | ✅ |

#### 16k Evaluation

| Step | LCBv5-16k (p@1) | LCBv6-16k (p@1) | Ready |
|------|-----------------|-----------------|-------|
| 10 | ✅ 10.2 | ✅ 12.1 | ✅ |
| 50 | ✅ 9.2 | ✅ 11.0 | ✅ |
| 80 | ✅ 8.0 | ✅ 10.3 | ✅ |
| 105 | ✅ 6.7 | ✅ 10.4 | ✅ |

### `kl0005`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.7 | ✅ 9.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.3 | ✅ 10.5 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.3 | ✅ 9.8 | ✅ |
| 105 | ❌ | ❌ | ❌ | ❌ | ❌ |

### `n12`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 9.3 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.2 | ✅ 11.2 | ✅ |
| 80 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 105 | ❌ | ❌ | ❌ | ❌ | ❌ |

### `n8`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.8 | ✅ 10.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 8.9 | ✅ 10.8 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.3 | ✅ 11.5 | ✅ |
| 105 | ❌ | ❌ | ❌ | ❌ | ❌ |

### `temp08`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.5 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.4 | ✅ 11.0 | ✅ |
| 80 | ✅ | ✅ | ✅ 7.9 | ✅ 9.7 | ✅ |
| 105 | ✅ | ✅ | ✅ 7.6 | ✅ 9.7 | ✅ |

### `temp12`
Full name: `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.8 | ✅ 10.2 | ✅ |
| 50 | ✅ | ✅ | ✅ 8.9 | ✅ 11.5 | ✅ |
| 80 | ✅ | ✅ | ✅ 7.3 | ✅ 10.5 | ✅ |
| 105 | ✅ | ✅ | ✅ 6.6 | ✅ 10.0 | ✅ |


## Method: ProbDisparity

### `baseline`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.8 | ✅ 10.3 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.1 | ✅ 11.4 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.3 | ✅ 12.0 | ✅ |
| 105 | ✅ | ✅ | ✅ 8.1 | ✅ 10.5 | ✅ |

#### 16k Evaluation

| Step | LCBv5-16k (p@1) | LCBv6-16k (p@1) | Ready |
|------|-----------------|-----------------|-------|
| 10 | ✅ 10.0 | ✅ 12.3 | ✅ |
| 50 | ✅ 10.2 | ✅ 10.6 | ✅ |
| 80 | ✅ 8.8 | ✅ 10.5 | ✅ |
| 105 | ✅ 7.9 | ✅ 10.1 | ✅ |

### `kl0005`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.8 | ✅ 10.9 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.1 | ✅ 11.5 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.5 | ✅ 10.9 | ✅ |
| 105 | ✅ | ✅ | ✅ 7.8 | ✅ 9.9 | ✅ |

### `n12`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 9.1 | ✅ 9.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.2 | ✅ 10.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.9 | ✅ 10.2 | ✅ |
| 105 | ✅ | ✅ | ✅ 8.3 | ✅ 10.3 | ✅ |

### `n8`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.6 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.0 | ✅ 11.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 9.1 | ✅ 10.8 | ✅ |
| 105 | ✅ | ✅ | ✅ 8.2 | ✅ 10.8 | ✅ |

### `temp08`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.2 | ✅ 10.0 | ✅ |
| 50 | ✅ | ✅ | ✅ 8.5 | ✅ 11.2 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.9 | ✅ 9.6 | ✅ |
| 105 | ✅ | ✅ | ✅ 8.4 | ✅ 10.0 | ✅ |

### `temp12`
Full name: `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.6 | ✅ 10.2 | ✅ |
| 50 | ✅ | ✅ | ✅ 10.0 | ✅ 11.4 | ✅ |
| 80 | ✅ | ✅ | ✅ 7.7 | ✅ 10.1 | ✅ |
| 105 | ✅ | ✅ | ✅ 7.1 | ✅ 10.1 | ✅ |


## Method: Pure-GRPO

### `baseline`
Full name: `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.0 | ✅ 10.0 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.4 | ✅ 10.0 | ✅ |
| 80 | ✅ | ✅ | ✅ 16.3 | ✅ 10.6 | ✅ |
| 105 | ✅ | ✅ | ✅ 9.4 | ✅ 11.0 | ✅ |

#### 16k Evaluation

| Step | LCBv5-16k (p@1) | LCBv6-16k (p@1) | Ready |
|------|-----------------|-----------------|-------|
| 10 | ✅ 15.7 | ✅ 11.7 | ✅ |
| 50 | ✅ 10.1 | ✅ 12.8 | ✅ |
| 80 | ✅ 17.9 | ✅ 11.6 | ✅ |
| 105 | ✅ 10.5 | ✅ 13.3 | ✅ |

### `clip_ratio_c10-ppo_epoch3`
Full name: `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 50 | ✅ | ❌ | ❌ | ❌ | 🔶 |
| 80 | ✅ | ❌ | ❌ | ❌ | 🔶 |
| 105 | ❌ | ❌ | ❌ | ❌ | ❌ |

### `n12`
Full name: `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 9.0 | ✅ 9.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.3 | ✅ 11.7 | ✅ |
| 80 | ✅ | ✅ | ✅ 9.4 | ✅ 10.9 | ✅ |
| 105 | ✅ | ✅ | ✅ 9.3 | ✅ 11.9 | ✅ |

### `n8`
Full name: `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.6 | ✅ 10.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 15.3 | ✅ 10.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 16.1 | ✅ 10.2 | ✅ |
| 105 | ✅ | ✅ | ✅ 10.1 | ✅ 10.6 | ✅ |


## Method: TokenEntropy

### `baseline`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.8 | ✅ 10.8 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.4 | ✅ 11.4 | ✅ |
| 80 | ✅ | ✅ | ✅ 13.2 | ✅ 10.8 | ✅ |
| 105 | ✅ | ✅ | ✅ 14.2 | ✅ 10.0 | ✅ |

#### 16k Evaluation

| Step | LCBv5-16k (p@1) | LCBv6-16k (p@1) | Ready |
|------|-----------------|-----------------|-------|
| 10 | ✅ 16.5 | ✅ 12.0 | ✅ |
| 50 | ✅ 16.4 | ✅ 11.1 | ✅ |
| 80 | ✅ 14.0 | ✅ 9.9 | ✅ |
| 105 | ✅ 13.3 | ✅ 11.0 | ✅ |

### `kl0005`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.1 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 15.8 | ✅ 10.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 14.7 | ✅ 11.2 | ✅ |
| 105 | ✅ | ✅ | ✅ 13.8 | ✅ 11.0 | ✅ |

### `n12`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 16.3 | ✅ 10.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.6 | ✅ 10.8 | ✅ |
| 80 | ✅ | ✅ | ✅ 14.7 | ✅ 10.1 | ✅ |
| 105 | ✅ | ✅ | ✅ 13.4 | ✅ 10.1 | ✅ |

### `n8`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.1 | ✅ 10.0 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.4 | ✅ 11.7 | ✅ |
| 80 | ✅ | ✅ | ✅ 15.4 | ✅ 10.7 | ✅ |
| 105 | ✅ | ✅ | ✅ 14.9 | ✅ 10.2 | ✅ |

### `temp08`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 14.7 | ✅ 10.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.3 | ✅ 10.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 15.5 | ✅ 11.4 | ✅ |
| 105 | ✅ | ✅ | ✅ 14.5 | ✅ 11.0 | ✅ |

### `temp12`
Full name: `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.2 | ✅ 9.1 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.9 | ✅ 10.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 14.1 | ✅ 11.3 | ✅ |
| 105 | ✅ | ✅ | ✅ 12.3 | ✅ 9.6 | ✅ |


## Method: TrajectoryEntropy

### `baseline`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 14.7 | ✅ 9.5 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.3 | ✅ 11.5 | ✅ |
| 80 | ✅ | ✅ | ✅ 7.4 | ✅ 11.0 | ✅ |
| 105 | ✅ | ✅ | ✅ 13.7 | ✅ 11.5 | ✅ |

#### 16k Evaluation

| Step | LCBv5-16k (p@1) | LCBv6-16k (p@1) | Ready |
|------|-----------------|-----------------|-------|
| 10 | ✅ 9.6 | ✅ 11.7 | ✅ |
| 50 | ✅ 16.6 | ✅ 11.7 | ✅ |
| 80 | ✅ 14.7 | ✅ 10.7 | ✅ |
| 105 | ✅ 12.9 | ✅ 10.7 | ✅ |

### `kl0005`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.4 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 15.9 | ✅ 11.3 | ✅ |
| 80 | ✅ | ✅ | ✅ 14.2 | ✅ 10.2 | ✅ |
| 105 | ✅ | ✅ | ✅ 13.5 | ✅ 9.7 | ✅ |

### `n12`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.7 | ✅ 10.8 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.1 | ✅ 11.2 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.4 | ✅ 10.9 | ✅ |
| 105 | ✅ | ✅ | ✅ 13.4 | ✅ 11.0 | ✅ |

### `n8`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 15.1 | ✅ 10.8 | ✅ |
| 50 | ✅ | ✅ | ✅ 16.1 | ✅ 11.9 | ✅ |
| 80 | ✅ | ✅ | ✅ 15.6 | ✅ 10.4 | ✅ |
| 105 | ✅ | ✅ | ❌ | ✅ 10.7 | 🔶 |

### `temp08`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.5 | ✅ 10.0 | ✅ |
| 50 | ✅ | ✅ | ✅ 9.3 | ✅ 11.0 | ✅ |
| 80 | ✅ | ✅ | ✅ 8.8 | ✅ 11.1 | ✅ |
| 105 | ✅ | ✅ | ✅ 8.2 | ✅ 11.0 | ✅ |

### `temp12`
Full name: `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12`

#### 8k Evaluation

| Step | AIME24 | AIME25 | LCBv5 (p@1) | LCBv6 (p@1) | Ready |
|------|--------|--------|-------------|-------------|-------|
| 10 | ✅ | ✅ | ✅ 8.6 | ✅ 10.4 | ✅ |
| 50 | ✅ | ✅ | ✅ 8.9 | ✅ 11.1 | ✅ |
| 80 | ✅ | ✅ | ✅ 13.9 | ✅ 10.5 | ✅ |
| 105 | ✅ | ✅ | ✅ 12.3 | ✅ 9.9 | ✅ |


## Summary: Fully Ready Experiments (all 4 steps)

An experiment is "fully ready (8k)" if steps 10,50,80,105 all have: AIME24 ✅ AIME25 ✅ LCBv5_8k ✅ LCBv6_8k ✅

| Experiment | Method | Variant | Steps Ready (8k) | Fully Ready |
|-----------|--------|---------|-----------------|-------------|
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | baseline | 10,50,80,105 | ✅ All 4 |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | kl0005 | 10,50,80 | 🔶 3/4 (10,50,80) |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | n12 | 10,50 | 🔶 2/4 (10,50) |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | n8 | 10,50,80 | 🔶 3/4 (10,50,80) |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | temp08 | 10,50,80,105 | ✅ All 4 |
| `Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-n...` | Intuitor | temp12 | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | baseline | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | kl0005 | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | n12 | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | n8 | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | temp08 | 10,50,80,105 | ✅ All 4 |
| `Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batc...` | ProbDisparity | temp12 | 10,50,80,105 | ✅ All 4 |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-n...` | Pure-GRPO | baseline | 10,50,80,105 | ✅ All 4 |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-n...` | Pure-GRPO | clip_ratio_c10-ppo_epoch3 | — | 🔶 0/4 () |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-n...` | Pure-GRPO | n12 | 10,50,80,105 | ✅ All 4 |
| `Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-n...` | Pure-GRPO | n8 | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | baseline | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | kl0005 | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | n12 | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | n8 | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | temp08 | 10,50,80,105 | ✅ All 4 |
| `Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch...` | TokenEntropy | temp12 | 10,50,80,105 | ✅ All 4 |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | baseline | 10,50,80,105 | ✅ All 4 |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | kl0005 | 10,50,80,105 | ✅ All 4 |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | n12 | 10,50,80,105 | ✅ All 4 |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | n8 | 10,50,80 | 🔶 3/4 (10,50,80) |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | temp08 | 10,50,80,105 | ✅ All 4 |
| `Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-...` | TrajectoryEntropy | temp12 | 10,50,80,105 | ✅ All 4 |

---
*Auto-generated by analysis script*