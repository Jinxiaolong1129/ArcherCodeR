# Intuitor Algorithm Integration Guide

## Overview

The Intuitor algorithm has been successfully integrated into the VERL framework. Intuitor is a novel reinforcement learning approach that uses the model's self-certainty (confidence in its own predictions) as an intrinsic reward signal, eliminating the need for external reward models.

## Key Features

- **Self-Supervised Learning**: Uses model's internal confidence as reward signal
- **No External Reward Model**: Eliminates dependency on separate reward models
- **Efficient Computation**: Only requires additional logits processing
- **GRPO-Compatible**: Uses GRPO-style advantage computation for stability

## Algorithm Details

### Self-Certainty Calculation

The self-certainty metric is computed as:
```
self_certainty = logsumexp(logits) - mean(logits)
```

This measures how confident the model is about its predictions. Higher values indicate higher confidence.

### Advantage Estimation

Intuitor follows these steps:
1. Compute token-level self-certainty from logits
2. Calculate sentence-level mean self-certainty
3. Use GRPO-style advantage computation with self-certainty as reward
4. Apply group normalization for stability

## Usage

### Configuration

To use Intuitor, set the advantage estimator in your configuration:

```yaml
algorithm:
  adv_estimator: intuitor
  norm_adv_by_std_in_grpo: true  # Enable advantage normalization
```

### Command Line

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    # ... other configurations
```

### Example Script

See `verl/examples/intuitor_demo.sh` for a complete example.

## Implementation Details

### Files Modified

1. **`verl/utils/torch_functional.py`**: Added `self_certainty_from_logits()` function
2. **`verl/trainer/ppo/core_algos.py`**: 
   - Added `AdvantageEstimator.INTUITOR` enum
   - Added `compute_intuitor_advantage()` function
3. **`verl/workers/actor/dp_actor.py`**: Modified to compute self-certainty
4. **`verl/workers/fsdp_workers.py`**: Updated to pass self-certainty data
5. **`verl/trainer/ppo/ray_trainer.py`**: Added Intuitor-specific handling

### Key Functions

- `self_certainty_from_logits(logits)`: Computes self-certainty from model logits
- `compute_intuitor_advantage()`: Registered advantage estimator function
- Modified `_forward_micro_batch()`: Now computes self-certainty alongside entropy

## Performance Considerations

- **Memory**: Minimal additional memory overhead (only self-certainty tensors)
- **Computation**: Efficient logits processing with optional torch.compile optimization
- **Scalability**: Works with FSDP, sequence parallelism, and other optimizations

## Comparison with Other Methods

| Method | External Reward | Computation Cost | Stability |
|--------|----------------|------------------|-----------|
| PPO+RM | Required | High | Good |
| GRPO | Required | Medium | Good |
| Intuitor | Not Required | Low | Good |

## Troubleshooting

### Common Issues

1. **Missing self_certaintys in batch**: Ensure compute_log_prob is called with `calculate_self_certainty=True`
2. **Shape mismatches**: Verify response_mask and self_certaintys have compatible shapes
3. **NaN values**: Check for proper masking and avoid division by zero

### Debug Tips

- Enable debug prints in `compute_intuitor_advantage()` to inspect values
- Verify self-certainty values are reasonable (typically in range [-10, 10])
- Check that response masks are correctly applied

## Future Enhancements

- [ ] Support for multi-turn conversations
- [ ] Integration with other advantage estimators
- [ ] Adaptive self-certainty thresholding
- [ ] Performance optimizations for large models

## References

- Original Intuitor paper: [Add reference when available]
- GRPO paper: [Group Relative Policy Optimization]
- VERL framework: [VERL documentation] 