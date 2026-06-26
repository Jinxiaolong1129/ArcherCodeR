# Bug: Length Penalty Broadcast Mismatch in Traj/Token Entropy

**Status:** Open
**Found:** 2026-06-18
**File:** `verl/trainer/ppo/ray_trainer.py`
**Affected methods:** `TRAJECTORY_ENTROPY` (line ~708), `TOKEN_ENTROPY` (line ~749)
**Not affected:** `INTUITOR` (Self-Certainty, line ~328)

## Summary

When the short length penalty is enabled, Trajectory Entropy and Token Entropy use `* response_mask_float` to broadcast the penalized sequence-level score to token level, while INTUITOR uses `expand_as`. This causes the GRPO score sum to be **length-dependent** for Traj/Token Entropy, creating a reverse incentive that makes responses **shorter** instead of longer.

## Root Cause

In `compute_grpo_outcome_advantage`, the first operation is:

```python
scores = token_level_rewards.sum(dim=-1)   # core_algos.py:211
```

The three methods broadcast `penalized_scores` (shape `[B]`) to `token_level_rewards` (shape `[B, T]`) differently:

### INTUITOR (correct)
```python
token_level_rewards = penalized_scores.unsqueeze(1).expand_as(self_certaintys)  # line 328
```
- Every position (including padding) gets the value.
- `sum(dim=-1)` = `penalized_score * T_full` where `T_full` is the tensor dimension (constant across all samples).
- Since the multiplier is constant, GRPO group normalization cancels it out. The length penalty signal is preserved.

### TRAJECTORY_ENTROPY / TOKEN_ENTROPY (buggy)
```python
token_level_rewards = penalized_scores.unsqueeze(1) * response_mask_float       # line 708, 749
```
- Only response positions get the value; padding is zeroed out.
- `sum(dim=-1)` = `penalized_score * L` where `L` is the actual response length (varies per sample).
- Since `penalized_score` is typically **negative** (mean log-prob for Traj Entropy, negative mean entropy for Token Entropy), longer responses get a **more negative** sum, making them rank lower in GRPO normalization.

## Example

For Trajectory Entropy, `seq_scores = mean(log_probs)` which is negative:

| Response | L | penalized_score | GRPO sum = score * L |
|----------|------|-----------------|----------------------|
| Short | 1000 | -1.3 (with LP) | -1300 |
| Long | 6000 | -0.5 (no LP) | -3000 |

After GRPO group normalization, the **short** response gets higher advantage despite the length penalty trying to discourage short responses. The `* L` scaling overwhelms the penalty.

## Impact

The length penalty ablation experiments for Token Entropy and Trajectory Entropy in the paper (Appendix G.4) show responses getting **shorter** with LP enabled, which is the opposite of the intended behavior. The INTUITOR (Self-Certainty) and Probability Disparity results are not affected if they use the `expand_as` path.

## Fix

Replace `* response_mask_float` with `expand_as` in the LP-enabled branches of `TRAJECTORY_ENTROPY` and `TOKEN_ENTROPY`:

```python
# Before (buggy):
token_level_rewards = penalized_scores.unsqueeze(1) * response_mask_float

# After (fixed):
token_level_rewards = penalized_scores.unsqueeze(1).expand_as(response_mask_float)
```

Alternatively, pass the sequence-level `penalized_scores` directly to a modified GRPO function that accepts sequence-level rewards, avoiding the broadcast entirely.

After fixing, the length penalty experiments for Traj Entropy and Token Entropy need to be re-run.

## Prob Disparity

Prob Disparity LP branch (line ~790) uses the same `* response_mask_float` pattern. However, its `seq_scores = mean(prob_disparity)` is **non-negative** (max_prob - second_max_prob >= 0), so the `* L` scaling does not reverse the incentive direction. It still distorts the penalty effect (longer responses get disproportionately higher GRPO scores), but the symptom is less severe than for Traj/Token Entropy where scores are negative. Should still be fixed for correctness.
