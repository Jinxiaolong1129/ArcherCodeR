# Unified scripts for Pure-GRPO and self-RL methods

This folder contains copied-and-modified scripts for:
- Pure-GRPO
- Intuitor
- Trajectory Entropy
- Token Entropy
- Prob Disparity

Unified settings (explicit):
- `actor_rollout_ref.actor.optim.warmup_style=constant`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=10`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.ppo_epochs=1`
- `trainer.total_epochs=2`
- `actor_rollout_ref.actor.clip_ratio_c=3.0`

Strictly unified checkpoint/eval settings:
- `trainer.save_freq=10`
- `trainer.test_freq=10`
- `trainer.resume_mode=auto`
- `+trainer.max_actor_ckpt_to_keep=60`

Parameter override support in each `*-unified.sh`:
- `EXP_SUFFIX`: append suffix to `exp_name`
- `TEMP_OVERRIDE`: override `temperature`
- `N_RESP_OVERRIDE`: override `n_resp_per_prompt`
- `USE_KL_LOSS_OVERRIDE`: override `use_kl_loss` (e.g. `True`)
- `KL_LOSS_COEF_OVERRIDE`: override `kl_loss_coef` (e.g. `0.001`)

Comparison runners:
- `run_compare_temp.sh`: run all 5 methods with temp `0.8` and `1.2`
- `run_compare_n.sh`: run all 5 methods with `n=8` and `n=12`
- `run_compare_kl.sh`: run all 5 methods with `kl_loss_coef=0.001` and `0.005`
