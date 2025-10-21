# DACE: Difficulty-Aware Certainty Exploration

## Overview

DACE (Difficulty-Aware Certainty Exploration) is an advanced reinforcement learning algorithm that adaptively balances exploration and exploitation based on task difficulty and model certainty.

## 🚧 Problem Motivation

### 1. Sparse and Coarse Rewards

In RLVF (Reinforcement Learning with Verifier Feedback), models typically receive **binary rewards** (correct = 1, incorrect = 0). However:

* Binary rewards don't distinguish between **good vs. inefficient** correct solutions
* They treat all wrong answers equally, offering no guidance on *how* to improve reasoning
* This causes inefficient learning because the model can't tell when to explore new reasoning paths vs. refine existing ones

### 2. Certainty as a Signal

The model's **self-certainty** (how confident it is in its generated answer) correlates with **task difficulty** and **solution quality**:

* On **hard problems**, low-certainty responses often reflect valuable exploration
* On **easy problems**, high-certainty responses tend to indicate efficient, high-quality reasoning

Thus, *certainty* can act as a continuous signal guiding whether to **explore** or **exploit**.

## ⚙️ DACE Method

### 1. Difficulty Estimation

The model estimates task difficulty *relative to its own performance*:

```
diff(x; π) = 1 - E_{y~π(·|x)} [verify(y)]
```

This is approximated by sampling multiple responses and measuring the **failure rate**.

* `diff = 0.0`: All responses correct → easy task
* `diff = 1.0`: All responses wrong → hard task

### 2. Certainty Metric

Certainty is computed as the **negative average log-probability** of the generated sequence:

```
C(y, x; π) = -1/|y| Σ_j log π(y_j | x, y_{<j})
```

* **Higher certainty** → model uses high-probability tokens → exploitation
* **Lower certainty** → model uses low-probability tokens → exploration

### 3. Adaptive Intrinsic Reward

DACE introduces an **adaptive intrinsic reward** term:

```
R_int(x, y; π) = α(x; π) · C(y, x; π)
```

where the adaptive coefficient is:

```
α(x; π) = α_scale · sgn(β_threshold - diff(x; π))
```

This means:

* For **difficult tasks** (`diff > β_threshold`): `α < 0` → encourages **exploration** (lower certainty gets higher reward)
* For **easy tasks** (`diff < β_threshold`): `α > 0` → encourages **exploitation** (higher certainty gets higher reward)

The total training objective combines external rewards (correctness) and this adaptive intrinsic reward:

```
max_π E[R_ext + R_int]
```

## 📊 Implementation Details

### File Changes

1. **`verl/trainer/ppo/core_algos.py`**
   - Added `DACE = "dace"` to `AdvantageEstimator` enum
   - Implemented `compute_dace_advantage()` function with full DACE algorithm

2. **`verl/trainer/ppo/ray_trainer.py`**
   - Added DACE handling in `compute_advantage()` function
   - Added DACE to list of critic-free estimators in `_validate_config()`
   - Added detailed logging for DACE metrics

3. **Training Script**
   - Created `run_DACE-Qwen2.5-1.5B-gsm8k-math.sh`

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | - | Set to `"dace"` to use DACE |
| `algorithm.dace_alpha_scale` | 0.1 | Scaling factor for intrinsic reward |
| `algorithm.dace_beta_threshold` | 0.5 | Difficulty threshold (0.0-1.0) |
| `algorithm.norm_adv_by_std_in_grpo` | True | Normalize advantages by std |
| `actor_rollout_ref.rollout.n` | 8 | Number of responses per prompt |

## 🚀 Usage

### Basic Usage

```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-gsm8k-math.sh
```

### Custom Configuration

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dace \
    algorithm.dace_alpha_scale=0.2 \
    algorithm.dace_beta_threshold=0.6 \
    actor_rollout_ref.rollout.n=8 \
    # ... other parameters
```

### Hyperparameter Tuning Guide

#### α_scale (Intrinsic Reward Scaling)

* **Small (0.05-0.1)**: Subtle intrinsic signal, primarily driven by external rewards
* **Medium (0.1-0.3)**: Balanced combination of external and intrinsic rewards
* **Large (0.3-0.5)**: Strong intrinsic signal, aggressive exploration/exploitation

**Recommendation**: Start with 0.1, increase if the model:
- Gets stuck in local optima (increase to 0.2-0.3)
- Shows poor sample efficiency on hard problems (increase to 0.2-0.3)

#### β_threshold (Difficulty Threshold)

* **Low (0.3-0.4)**: More tasks treated as "easy" → more exploitation
* **Medium (0.5-0.6)**: Balanced threshold
* **High (0.6-0.8)**: More tasks treated as "hard" → more exploration

**Recommendation**: Start with 0.5, adjust based on:
- Dataset difficulty: Harder datasets → higher threshold (0.6-0.7)
- Training phase: Early training → higher threshold, later → lower threshold

#### Number of Responses (n)

* **Small (4-8)**: Faster training, less accurate difficulty estimation
* **Medium (8-16)**: Good balance
* **Large (16-32)**: More accurate difficulty estimation, slower training

**Recommendation**: Use 8-16 for most cases. Use 16+ only if:
- You have ample compute resources
- Dataset has highly variable difficulty
- Difficulty estimation accuracy is critical

## 📈 Expected Behavior

### Training Logs

You should see logs like:

```
-------------------------------- This is DACE --------------------------------
DACE Hyperparameters:
  α_scale (intrinsic reward scaling): 0.1
  β_threshold (difficulty threshold): 0.5
data.batch['token_level_rewards'].shape: torch.Size([1024, 512])
data.batch['old_log_probs'].shape: torch.Size([1024, 512])
External reward range: [0.0000, 1.0000]
Certainty range: [0.5234, 3.8471]
Advantage range: [2.1543, -1.8765]
-------------------------------- End of DACE --------------------------------
```

### Metrics to Monitor

1. **External Rewards**: Should increase over time (model getting more answers correct)
2. **Certainty Distribution**: 
   - For hard tasks: Lower certainty should correlate with higher advantages
   - For easy tasks: Higher certainty should correlate with higher advantages
3. **Difficulty Distribution**: Observe how tasks are classified as easy vs hard
4. **Advantage Distribution**: Should show clear separation between good/bad responses

## 🔬 Comparison with Other Methods

| Method | External Reward | Intrinsic Reward | Adaptive |
|--------|----------------|------------------|----------|
| GRPO | ✅ Binary | ❌ None | ❌ No |
| INTUITOR | ❌ None | ✅ Certainty | ❌ No |
| INTUITOR_SELECTIVE | ✅ Binary | ✅ Certainty | ⚠️ Partial |
| **DACE** | ✅ Binary | ✅ Certainty | ✅ Full |

### Advantages of DACE

1. **Adaptive**: Automatically adjusts exploration/exploitation based on task difficulty
2. **Efficient**: Uses both external (correctness) and intrinsic (certainty) signals
3. **Task-aware**: Treats different tasks differently based on difficulty
4. **No additional models**: Doesn't require external reward models or verifiers beyond correctness

## 🔧 Troubleshooting

### Issue: Training is unstable

**Solution**: 
- Reduce `alpha_scale` (try 0.05 or 0.1)
- Increase `rollout.n` for better difficulty estimation
- Enable advantage normalization: `algorithm.norm_adv_by_std_in_grpo=True`

### Issue: Model not exploring enough on hard tasks

**Solution**:
- Increase `alpha_scale` (try 0.2-0.3)
- Increase `beta_threshold` (try 0.6-0.7)
- Verify that hard tasks have `diff > beta_threshold`

### Issue: Model not exploiting enough on easy tasks

**Solution**:
- Increase `alpha_scale` (try 0.2-0.3)
- Decrease `beta_threshold` (try 0.3-0.4)
- Verify that easy tasks have `diff < beta_threshold`

### Issue: All tasks classified as same difficulty

**Solution**:
- Increase `rollout.n` (more responses for better statistics)
- Check dataset balance (ensure mix of easy and hard problems)
- Adjust `beta_threshold` to better match your dataset

## 📚 References

DACE is based on concepts from:
- Reinforcement Learning with Verifier Feedback (RLVF)
- Intrinsic Motivation in RL
- Adaptive Exploration Strategies
- GRPO (Group Relative Policy Optimization)

## 🤝 Contributing

If you encounter issues or have suggestions for improving DACE:
1. Check the troubleshooting section above
2. Review training logs for unexpected behavior
3. Experiment with different hyperparameters
4. Document your findings for future reference

## 📝 Citation

If you use DACE in your research, please cite:

```bibtex
@article{dace2024,
  title={DACE: Difficulty-Aware Certainty Exploration for Efficient Reinforcement Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

