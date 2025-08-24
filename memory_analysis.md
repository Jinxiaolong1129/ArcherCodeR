# Intuitor vs PPO Memory Analysis

## Configuration
- Model: Qwen2.5-1.5B (vocab_size = 151,936)
- Batch size: 64
- Sequence length: 2K prompt + 4K response = 6K tokens
- Precision: bfloat16 (2 bytes) for storage, float32 (4 bytes) for computation

## Memory Breakdown

### Regular PPO Memory Usage
```
Component                    Size                Memory
Model parameters            1.5B × 2 bytes     = 3GB
Model activations           64 × 6K × 1536 × 2 = 1.2GB  
Log probabilities           64 × 6K × 2        = 0.8MB
Gradients (parameters)      1.5B × 2 bytes     = 3GB
Other (optimizer states)    ~2×params          = 6GB
----------------------------------------
Total Regular PPO                              ≈ 13.2GB
```

### Intuitor Additional Memory
```
Component                    Size                Memory
Full logits storage         64 × 6K × 152K × 4 = 234GB
Logits gradients           64 × 6K × 152K × 4 = 234GB  
Self-certainty temps       64 × 6K × 4 × 3    = 4.6MB
----------------------------------------
Additional for Intuitor                        ≈ 468GB
Total Intuitor Memory                          ≈ 481GB
```

## Memory Scaling Factors

| Sequence Length | Regular PPO | Intuitor    | Ratio |
|----------------|-------------|-------------|-------|
| 2K tokens      | ~8GB        | ~160GB      | 20x   |
| 4K tokens      | ~13GB       | ~320GB      | 25x   |
| 6K tokens      | ~18GB       | ~480GB      | 27x   |
| 8K tokens      | ~23GB       | ~640GB      | 28x   |

## Why Such High Memory Usage?

1. **Vocabulary Size Impact**: 152K vocab means each token position needs 152K logit values
2. **No Compression**: Unlike log_probs which compress to single values, logits keep full distribution
3. **Gradient Storage**: Backprop through logits requires storing gradients for entire vocab dimension
4. **Sequence Length**: Memory scales linearly with sequence length

## Optimization Strategies

### 1. Reduce Sequence Length (Your Approach)
```bash
# Original: 2K + 16K = 18K tokens → ~1.4TB memory
max_response_length=$((1024 * 2))    # 2K response
v_max_response_length=$((1024 * 4))  # 4K validation

# New: 2K + 2K = 4K tokens → ~160GB memory
# Reduction: ~90% memory savings
```

### 2. Gradient Checkpointing for Logits
```python
# Could implement chunked self-certainty computation
def chunked_self_certainty(logits, chunk_size=1024):
    chunks = []
    for i in range(0, logits.size(-1), chunk_size):
        chunk = logits[..., i:i+chunk_size]
        chunk_certainty = torch.logsumexp(chunk, dim=-1) - chunk.mean(dim=-1)
        chunks.append(chunk_certainty)
    return torch.stack(chunks, dim=-1).mean(dim=-1)
```

### 3. Mixed Precision Optimization
```python
# Keep logits in bfloat16, only convert to float32 for computation
with torch.cuda.amp.autocast():
    self_certainty = compute_self_certainty(logits.float()).half()
```

## Recommended Settings for 8×80GB GPUs

```bash
# Conservative (fits in memory)
max_response_length=1024      # 1K response
train_prompt_bsz=16          # Small batch
n_resp_per_prompt=4          # Fewer responses

# Aggressive (maximum utilization)  
max_response_length=2048      # 2K response
train_prompt_bsz=32          # Medium batch
n_resp_per_prompt=8          # Moderate responses
``` 