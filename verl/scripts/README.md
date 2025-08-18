# INTUITOR Training Scripts for ArcherCodeR

This directory contains training scripts for the INTUITOR algorithm implementation in ArcherCodeR/VERL.

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default configuration
./run_intuitor.sh

# Run with specific task configuration
./run_intuitor.sh math_gsm8k
./run_intuitor.sh code_humaneval
./run_intuitor.sh reasoning_arc
```

### Simple Math Training

```bash
# Direct math training (similar to original math_intuitor.sh)
./math_intuitor_archercoder.sh
```

## 📁 Available Scripts

### 1. `run_intuitor.sh` - Main Training Script
**Recommended for most users**

Features:
- ✅ Multiple predefined configurations
- ✅ Environment validation
- ✅ Colored output and progress tracking
- ✅ Automatic log management
- ✅ Error handling and validation

**Usage:**
```bash
./run_intuitor.sh [config_name]
```

**Available Configurations:**
- `math_gsm8k` - GSM8K math problems
- `math_math500` - MATH500 dataset
- `code_humaneval` - HumanEval code generation
- `reasoning_arc` - ARC reasoning tasks
- `default` - General purpose configuration

### 2. `math_intuitor_archercoder.sh` - Simple Math Script
**Direct port of original math_intuitor.sh**

Features:
- ✅ Simple, direct configuration
- ✅ Matches original script structure
- ✅ Good for reproducing original results

### 3. `run_intuitor_training.sh` - Advanced Configuration
**For advanced users who need full control**

Features:
- ✅ Environment variable configuration
- ✅ Extensive parameter customization
- ✅ Production-ready setup

## 🔧 Configuration

### Environment Variables

You can customize training by setting environment variables:

```bash
# Model and data
export MODEL_PATH="Qwen/Qwen2.5-7B"
export TRAIN_DATA="$HOME/data/my_dataset/train.parquet"
export VAL_DATA="$HOME/data/my_dataset/test.parquet"

# Training parameters
export TRAIN_BATCH_SIZE=256
export LEARNING_RATE=1e-5
export TOTAL_EPOCHS=3

# Hardware settings
export N_GPUS=16
export N_NODES=2

# Run training
./run_intuitor.sh custom_config
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen2.5-3B` | HuggingFace model path |
| `TRAIN_BATCH_SIZE` | `128` | Training batch size |
| `LEARNING_RATE` | `3e-6` | Learning rate |
| `ROLLOUT_N` | `8` | Number of responses per prompt |
| `MAX_PROMPT_LENGTH` | `512` | Maximum prompt length |
| `MAX_RESPONSE_LENGTH` | `2048` | Maximum response length |
| `N_GPUS` | `8` | GPUs per node |
| `N_NODES` | `1` | Number of nodes |

## 📊 Monitoring

### Weights & Biases Integration

The scripts automatically log to W&B. Set your API key:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

### Log Files

Training logs are automatically saved to:
```
verl/logs/verl_${EXPERIMENT_NAME}_${TIMESTAMP}.log
```

### Key Metrics to Monitor

When using INTUITOR, pay attention to these metrics:

- `intuitor/self_certainty/mean` - Average self-certainty
- `intuitor/self_certainty/std` - Self-certainty variance
- `actor/entropy` - Model entropy
- `critic/advantages/mean` - Advantage estimates
- `perf/throughput` - Training throughput

## 🎯 Algorithm Details

### INTUITOR Overview

INTUITOR is a critic-free reinforcement learning algorithm that uses the model's self-certainty as an intrinsic reward signal:

1. **Self-Certainty Calculation**: `logsumexp(logits) - mean(logits)`
2. **Sentence-Level Aggregation**: Average self-certainty across response tokens
3. **GRPO-Style Advantages**: Group-based advantage estimation
4. **Policy Update**: Standard PPO update with self-certainty rewards

### Key Features

- ✅ **No External Reward Model**: Uses intrinsic self-certainty
- ✅ **Critic-Free**: No value function needed
- ✅ **Efficient**: Lower computational overhead
- ✅ **Self-Supervised**: Model learns to assess its own quality

## 🛠 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch sizes
   export PPO_MICRO_BATCH_SIZE_PER_GPU=2
   export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
   ```

2. **Data Not Found**
   ```bash
   # Check data paths
   ls -la $HOME/data/math/
   ```

3. **Model Download Issues**
   ```bash
   # Pre-download models
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-3B')"
   ```

### Performance Optimization

1. **Enable Remove Padding** (default: enabled)
   ```bash
   export USE_REMOVE_PADDING=true
   ```

2. **Gradient Checkpointing** (default: enabled)
   ```bash
   export ENABLE_GRADIENT_CHECKPOINTING=true
   ```

3. **Memory Optimization**
   ```bash
   export GPU_MEMORY_UTILIZATION=0.9
   ```

## 📈 Expected Results

### Training Progress

You should see:
- Decreasing policy loss
- Stable self-certainty values
- Improving validation metrics
- Consistent throughput

### Typical Timeline

- **Setup**: 2-5 minutes
- **Training**: 30-120 minutes per epoch (depends on data size)
- **Validation**: 5-15 minutes per validation run

## 🤝 Contributing

To add new configurations:

1. Edit `run_intuitor.sh`
2. Add new case in the configuration section
3. Test with your dataset
4. Submit a PR

## 📚 References

- [INTUITOR Paper](link-to-paper)
- [VERL Documentation](https://github.com/volcengine/verl)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## 📄 License

This project follows the same license as ArcherCodeR and VERL. 