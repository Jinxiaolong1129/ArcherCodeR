# Self-RL Training Methods Implementation

This document describes the implementation of four self-RL (self-rewarding reinforcement learning) training methods for comparing different intrinsic reward signals.

## Overview

All four methods follow the GRPO-style advantage computation pattern, where:
1. A scalar reward is computed for each response based on model confidence/certainty
2. Advantages are normalized within each prompt group using GRPO's group normalization
3. The advantages are broadcast to token-level for PPO training

## Implemented Methods

### 1. Self-Certainty (INTUITOR) ✅ Already Implemented

**Formula**: `r(x, y) = 1/|y| * Σ D_KL(U||π_θ(·|x, y_<t))`

**Implementation**: 
- Computed in `verl/utils/torch_functional.py::self_certainty_from_logits()`
- Formula: `logsumexp(logits) - mean(logits)`
- **Physical Meaning**: Measures how "peaked" the probability distribution is
  - Higher value = more confident (peaked distribution)
  - Lower value = less confident (flat distribution)

**Advantage Estimator**: `AdvantageEstimator.INTUITOR`

**Key Files**:
- `verl/utils/torch_functional.py`: `self_certainty_from_logits()`
- `verl/trainer/ppo/core_algos.py`: `compute_intuitor_advantage()`
- `verl/trainer/ppo/ray_trainer.py`: INTUITOR case in `compute_advantage()`

---

### 2. Trajectory-Level Entropy ✨ NEW

**Formula**: `r(x, y) = 1/|y| * Σ log π_θ(y_t|x, y_<t)`

**Implementation**:
- Uses `old_log_probs` directly (already computed during forward pass)
- Computes average log probability across the sequence
- **Physical Meaning**: Average likelihood of the generated trajectory
  - Higher value (closer to 0) = more confident trajectory
  - Lower value (more negative) = less confident trajectory

**Advantage Estimator**: `AdvantageEstimator.TRAJECTORY_ENTROPY`

**Key Files**:
- `verl/trainer/ppo/core_algos.py`: `compute_trajectory_entropy_advantage()`
- `verl/trainer/ppo/ray_trainer.py`: TRAJECTORY_ENTROPY case in `compute_advantage()`

**Data Requirements**:
- `old_log_probs`: Token-level log probabilities (already computed)
- `response_mask`: Response token mask

---

### 3. Token-Level Entropy ✨ NEW

**Formula**: `r(x, y) = -1/|y| * Σ H(π_θ(·|x, y_<t))`

**Implementation**:
- Computed in `verl/utils/torch_functional.py::entropy_from_logits()` (already exists)
- Uses negative average entropy as reward
- **Physical Meaning**: Negative average entropy of token distributions
  - Higher reward (less negative) = lower entropy = more confident
  - Lower reward (more negative) = higher entropy = less confident

**Advantage Estimator**: `AdvantageEstimator.TOKEN_ENTROPY`

**Key Files**:
- `verl/utils/torch_functional.py`: `entropy_from_logits()` (already exists)
- `verl/trainer/ppo/core_algos.py`: `compute_token_entropy_advantage()`
- `verl/trainer/ppo/ray_trainer.py`: TOKEN_ENTROPY case in `compute_advantage()`

**Data Requirements**:
- `entropys`: Token-level entropy values (already computed)
- `response_mask`: Response token mask

---

### 4. Probability Disparity ✨ NEW

**Formula**: `r(x, y) = 1/M * Σ [max π_θ(a_t|...) - second_max π_θ(a_t|...)]`

**Implementation**:
- Computed in `verl/utils/torch_functional.py::prob_disparity_from_logits()`
- Formula: `top1_prob - top2_prob` for each token
- **Physical Meaning**: Gap between top-1 and top-2 probabilities
  - Higher value = larger gap = more confident (clear winner)
  - Lower value = smaller gap = less confident (close competition)

**Advantage Estimator**: `AdvantageEstimator.PROB_DISPARITY`

**Key Files**:
- `verl/utils/torch_functional.py`: `prob_disparity_from_logits()`
- `verl/trainer/ppo/core_algos.py`: `compute_prob_disparity_advantage()`
- `verl/trainer/ppo/ray_trainer.py`: PROB_DISPARITY case in `compute_advantage()`
- `verl/workers/actor/dp_actor.py`: Added `calculate_prob_disparity` parameter
- `verl/workers/fsdp_workers.py`: Updated to compute and return `prob_disparitys`

**Data Requirements**:
- `prob_disparitys`: Token-level probability disparity values (newly computed)
- `response_mask`: Response token mask

---

## Architecture Changes

### 1. Core Algorithm Functions (`verl/trainer/ppo/core_algos.py`)

Added three new registered advantage estimator functions:
- `compute_trajectory_entropy_advantage()` - Uses old_log_probs
- `compute_token_entropy_advantage()` - Uses entropys
- `compute_prob_disparity_advantage()` - Uses prob_disparitys

All follow the same pattern:
```python
@register_adv_est(AdvantageEstimator.METHOD_NAME)
def compute_method_advantage(
    metric_tensor: torch.Tensor,  # The intrinsic reward signal
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    # 1. Compute sentence-wise mean of the metric
    # 2. Use GRPO-style group normalization
    # 3. Broadcast to token-level
    # 4. Return advantages and returns
```

### 2. Utility Functions (`verl/utils/torch_functional.py`)

Added new function:
- `prob_disparity_from_logits()` - Computes top1_prob - top2_prob

### 3. Actor Worker (`verl/workers/actor/dp_actor.py`)

Updated `_forward_micro_batch()` and `compute_log_prob()`:
- Added `calculate_prob_disparity` parameter
- Computes `prob_disparity` from logits when requested
- Returns 4-tuple: `(entropy, log_probs, self_certainty, prob_disparity)`

### 4. FSDP Worker (`verl/workers/fsdp_workers.py`)

Updated `compute_log_prob()`:
- Calls actor with `calculate_prob_disparity=True`
- Returns `prob_disparitys` in DataProto

### 5. Ray Trainer (`verl/trainer/ppo/ray_trainer.py`)

Added three new cases in `compute_advantage()`:
- `AdvantageEstimator.TRAJECTORY_ENTROPY`
- `AdvantageEstimator.TOKEN_ENTROPY`
- `AdvantageEstimator.PROB_DISPARITY`

Updated `use_critic` check to include the three new methods (they don't use critic).

---

## Usage

To use any of these methods, set the advantage estimator in your config:

```yaml
algorithm:
  adv_estimator: "trajectory_entropy"  # or "token_entropy" or "prob_disparity"
  norm_adv_by_std_in_grpo: true  # Whether to normalize by std in GRPO
```

---

## Comparison of Methods

| Method | Input Data | Computation Cost | Meaning |
|--------|-----------|------------------|---------|
| **Self-Certainty** | logits (full vocab) | Medium | KL divergence from uniform |
| **Trajectory Entropy** | log_probs (selected tokens) | Low | Average log probability |
| **Token Entropy** | logits (full vocab) | Medium | Negative average entropy |
| **Prob Disparity** | logits (full vocab) | Medium | Top-1 vs Top-2 gap |

### When to Use Each Method:

1. **Self-Certainty (INTUITOR)**: 
   - Best for: General confidence measurement
   - Captures: How peaked the distribution is vs uniform

2. **Trajectory-Level Entropy**:
   - Best for: Fastest computation, uses already-computed log_probs
   - Captures: Overall sequence likelihood

3. **Token-Level Entropy**:
   - Best for: Measuring distribution uncertainty
   - Captures: How spread out the probability mass is

4. **Probability Disparity**:
   - Best for: Measuring decision confidence
   - Captures: How clear the top choice is

---

## Implementation Notes

### Consistency with INTUITOR

All three new methods follow the exact same pattern as INTUITOR:
1. Compute token-level metric from model outputs
2. Average across response length (sentence-wise mean)
3. Apply GRPO-style group normalization within each prompt
4. Broadcast back to token-level for PPO training

### Numerical Stability

All methods include:
- `epsilon=1e-6` for division stability
- Proper device placement (tensors on same device)
- Masked operations to ignore padding tokens

### Logging

All methods include detailed logging:
- Input data shapes and statistics
- Computed metric ranges and distributions
- Advantage statistics (positive/negative split)

---

## Testing

To verify the implementation:

1. **Check data flow**: Ensure all required tensors are computed and passed correctly
2. **Verify shapes**: All tensors should have shape `(batch_size, response_length)`
3. **Check normalization**: Advantages should be normalized within each prompt group
4. **Monitor metrics**: Log the intrinsic reward values to ensure they're reasonable

Example test:
```python
# In your training config
algorithm:
  adv_estimator: "token_entropy"
  norm_adv_by_std_in_grpo: true

# Monitor these metrics during training:
# - token_entropy_reward/mean
# - token_entropy_reward/std
# - advantages/positive_ratio
```

---

## Future Extensions

Potential improvements:
1. **Hybrid methods**: Combine multiple intrinsic rewards with learnable weights
2. **Adaptive scaling**: Automatically adjust the importance of intrinsic rewards
3. **Task-specific tuning**: Different methods may work better for different tasks

---

## References

- INTUITOR (Self-Certainty): Uses KL divergence from uniform distribution
- Trajectory Entropy: Based on sequence log probability
- Token Entropy: Based on Shannon entropy of distributions
- Probability Disparity: Based on margin between top choices

