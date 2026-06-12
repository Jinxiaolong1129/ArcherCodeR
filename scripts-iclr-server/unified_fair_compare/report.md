# Unified Fair-Comparison Report (Pure-GRPO + 4 Self-RL Methods)

## 1. Goal and Scope

This report documents:
- The key parameter differences in the original scripts.
- What was modified to build a fair-comparison script suite.
- What was added later for parameter sweeps (`temp`, `n`, `kl`).
- Current status of throughput-related tuning and practical notes.

Scope of methods:
- Pure-GRPO
- Intuitor
- Trajectory Entropy
- Token Entropy
- Prob Disparity

All new scripts are under:
- `scripts-iclr-server/unified_fair_compare/`


## 2. Original Script Baseline (Before Unification)

Reference scripts analyzed:
- `scripts-iclr-server/train/run_Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl.sh`
- `scripts-iclr-server/intuitor/intuitor_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh`
- `scripts-iclr-server/trajectory_entropy/trajectory_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh`
- `scripts-iclr-server/token_entropy/token_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh`
- `scripts-iclr-server/prob_disparity/prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh`

### 2.1 Core differences that affected fairness

| Item | Pure-GRPO (original) | Self-RL variants (typical original) | Fairness impact |
|---|---|---|---|
| Learning rate | `1e-6` | mostly `3e-6` | Different optimization speed/trajectory |
| Warmup | constant-style via `lr_warmup_steps=10` | cosine-style via `warmup_style=cosine` + `lr_warmup_steps_ratio=0.1` | Different LR schedule shape |
| PPO epochs | `3` | `1` | Different update intensity per rollout |
| `clip_ratio_c` | explicitly `10.0` | often not explicit (default usually `3.0`) | Different negative-adv clipping behavior |
| `trainer.total_epochs` | `1` | mixed (`1` or `2`) | Different total training budget |
| Save/checkpoint policy | mixed (`save_freq`, `resume_mode`, `max_actor_ckpt_to_keep`) | mixed | Different storage/recovery cadence |


## 3. What Was Created

A new folder was created:
- `scripts-iclr-server/unified_fair_compare/`

Main method scripts:
- `run_Pure-GRPO-unified.sh`
- `run_intuitor-unified.sh`
- `run_trajectory_entropy-unified.sh`
- `run_token_entropy-unified.sh`
- `run_prob_disparity-unified.sh`

Sweep runners:
- `run_compare_temp.sh`
- `run_compare_n.sh`
- `run_compare_kl.sh`

Documentation:
- `README.md`


## 4. Unified Settings Applied Across 5 Method Scripts

The following were explicitly aligned:
- `actor_rollout_ref.actor.optim.warmup_style=constant`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=10`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.ppo_epochs=1`
- `trainer.total_epochs=2`
- `actor_rollout_ref.actor.clip_ratio_c=3.0`

Checkpoint/eval cadence was also unified:
- `trainer.save_freq=10`
- `trainer.test_freq=10`
- `trainer.resume_mode=auto`
- `+trainer.max_actor_ckpt_to_keep=60`


## 5. Parameter-Override Mechanism Added

To avoid cloning dozens of near-duplicate scripts, all 5 unified scripts support runtime overrides:
- `EXP_SUFFIX` (append to experiment name)
- `TEMP_OVERRIDE`
- `N_RESP_OVERRIDE`
- `USE_KL_LOSS_OVERRIDE`
- `KL_LOSS_COEF_OVERRIDE`

This allows controlled ablation while keeping the base script stable.


## 6. Sweep Scripts and Their Exact Coverage

### 6.1 Temperature sweep
- Script: `run_compare_temp.sh`
- Values: `0.8`, `1.2`
- Injected variables:
  - `EXP_SUFFIX=temp${temp}`
  - `TEMP_OVERRIDE=${temp}`
- Coverage: all 5 methods

### 6.2 Response-count (`n`) sweep
- Script: `run_compare_n.sh`
- Values: `8`, `12`
- Injected variables:
  - `EXP_SUFFIX=n${n}`
  - `N_RESP_OVERRIDE=${n}`
- Coverage: all 5 methods

### 6.3 KL sweep
- Script: `run_compare_kl.sh`
- Values: `0.001`, `0.005`
- Injected variables:
  - `EXP_SUFFIX=kl${coef_tag}`
  - `USE_KL_LOSS_OVERRIDE=True`
  - `KL_LOSS_COEF_OVERRIDE=${kl_coef}`
- Coverage: all 5 methods


## 7. Save Path Convention

All unified scripts use:
- `CKPTS_DIR=./output/${project_name}/${exp_name}`

With `EXP_SUFFIX` set, each run is separated automatically:
- Example:
  - `./output/ArcherCodeR/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp0.8/`
  - `./output/ArcherCodeR/Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12/`
  - `./output/ArcherCodeR/Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005/`


## 8. Throughput/Runtime Follow-up Change

After observing long step time, one throughput-oriented change was made in:
- `run_Pure-GRPO-unified.sh`

Change:
- `actor_rollout_ref.rollout.max_num_batched_tokens` was changed from:
  - `$((max_prompt_length + v_max_response_length))` (i.e., `10240`)
  - to fixed `65536`

Rationale:
- Increase vLLM token scheduling capacity per dispatch round.
- Improve rollout throughput for long-sequence generation without changing learning objective.

Note:
- This change is currently applied to `Pure-GRPO-unified` only.
- The 4 self-RL unified scripts still use:
  - `actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + v_max_response_length))`


## 9. Current Consistency Summary

### 9.1 Aligned (all 5 unified scripts)
- LR / warmup policy / PPO epochs / total epochs / clip ratio C
- Save/test/resume/checkpoint-keep policy
- Base prompt/response length (`2k` / `8k`)
- Base batching (`train_prompt_bsz=64`, `n_resp_per_prompt=16`) unless overridden by sweeps

### 9.2 Intentionally configurable
- `temperature`
- `n_resp_per_prompt`
- KL enable + KL coefficient

### 9.3 Current non-uniform item
- `max_num_batched_tokens` is tuned to `65536` only in `run_Pure-GRPO-unified.sh`


## 10. Reproduction / Usage Commands

Run one unified method script:
- `bash scripts-iclr-server/unified_fair_compare/run_Pure-GRPO-unified.sh`

Run parameter sweeps:
- `bash scripts-iclr-server/unified_fair_compare/run_compare_temp.sh`
- `bash scripts-iclr-server/unified_fair_compare/run_compare_n.sh`
- `bash scripts-iclr-server/unified_fair_compare/run_compare_kl.sh`

Run with additional Hydra args passthrough:
- `bash scripts-iclr-server/unified_fair_compare/run_compare_temp.sh trainer.total_epochs=1`


## 11. Practical Next Steps

For strict cross-method runtime fairness, decide whether to:
- Keep `max_num_batched_tokens` identical across all 5 scripts (recommended for pure fairness), or
- Tune each method separately for best throughput (recommended for performance-oriented benchmarking).

If fairness is the top priority, align this field across all scripts before final reporting.

