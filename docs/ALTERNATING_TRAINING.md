# 🔄 Alternating Training System

Enhanced alternating training system that supports dynamic switching between different RL algorithms (Intuitor, GRPO) with configurable parameters and KL loss modes.

## 🚀 Features

### ✨ Core Features
- **Dynamic Algorithm Switching**: Switch between Intuitor and GRPO without process restart
- **Configurable KL Loss**: Support for different KL loss coefficients (no-kl, kl005, kl01, kl02, kl05)
- **Flexible Output Paths**: Auto-generated paths with key parameters
- **Test Mode**: Quick testing with 1-step switching and limited datasets
- **Wandb Integration**: Configurable project names and experiment tracking

### 🎯 Algorithm Support
- **Intuitor**: Self-certainty based reward optimization
- **GRPO**: Group preference optimization with token entropy separation

## 📋 Usage

### Basic Usage

```bash
# Default training (50 steps per phase, no KL)
./scripts/train/run_alternating_unified.sh

# Custom configuration
./scripts/train/run_alternating_unified.sh \
    --steps-per-phase 25 \
    --kl-mode kl005 \
    --project-name "MyProject" \
    --total-epochs 6
```

### Test Mode

```bash
# Quick test (1-step switching, 1000 samples)
./scripts/train/run_alternating_unified.sh --test-mode

# Test with custom parameters
./scripts/train/run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 500 \
    --project-name "QuickTest"
```

### KL Loss Modes

```bash
# No KL loss (default)
./scripts/train/run_alternating_unified.sh --kl-mode no-kl

# KL coefficient = 0.05
./scripts/train/run_alternating_unified.sh --kl-mode kl005

# KL coefficient = 0.1
./scripts/train/run_alternating_unified.sh --kl-mode kl01
```

## 🛠️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithms` | Comma-separated algorithm list | `intuitor,grpo` |
| `--steps-per-phase` | Steps per algorithm phase | `50` |
| `--start-with` | Starting algorithm | `intuitor` |
| `--total-epochs` | Total training epochs | `4` |
| `--project-name` | Wandb project name | `ArcherCodeR` |
| `--kl-mode` | KL loss mode | `no-kl` |
| `--test-mode` | Enable test mode | `false` |
| `--dataset-limit` | Limit dataset samples | `null` |
| `--output-dir` | Custom output directory | auto-generated |
| `--config` | Configuration name | `alternating_official` |

## 📁 Output Directory Structure

The system automatically generates descriptive output directories:

```
./output/{PROJECT_NAME}/Alternating-{ALGORITHMS}-{KL_MODE}-steps{STEPS}-epochs{EPOCHS}-{TIMESTAMP}/
```

Examples:
- `./output/ArcherCodeR/Alternating-intuitor-grpo-no-kl-steps50-epochs4-20250915-143022/`
- `./output/MyProject/Test-intuitor-grpo-kl005-steps1-limit1000-20250915-143022/`

## ⚙️ Configuration Files

### Main Configuration: `alternating_official.yaml`
- Full production configuration
- Algorithm-specific parameters
- KL mode configurations
- Complete model and training settings

### Test Configuration: `alternating_test.yaml`
- Optimized for quick testing
- Reduced batch sizes and token lengths
- Limited epochs and dataset size
- 1-step algorithm switching

## 🧪 Testing

### Quick Test Script
```bash
./scripts/train/test_alternating.sh
```

### Example Scenarios
```bash
./scripts/train/examples_alternating.sh
```

Available examples:
1. **Quick Test**: 1-step switching, 1000 samples
2. **Standard Training**: 50-step phases, no KL
3. **KL Loss Training**: 25-step phases, KL=0.05
4. **Custom Project**: Different project name
5. **GRPO First**: Start with GRPO instead of Intuitor

## 📊 KL Loss Configuration

### Supported KL Modes

| Mode | KL Coefficient | Use Case |
|------|----------------|----------|
| `no-kl` | 0.0 | No KL regularization |
| `kl005` | 0.05 | Light regularization |
| `kl01` | 0.1 | Medium regularization |
| `kl02` | 0.2 | Strong regularization |
| `kl05` | 0.5 | Very strong regularization |

### Algorithm-Specific KL Settings

Each algorithm can have different KL configurations:

```yaml
kl_configs:
  kl005:
    intuitor:
      actor:
        use_kl_loss: true
        kl_loss_coef: 0.05
      algorithm:
        use_kl_in_reward: true
        kl_ctrl:
          kl_coef: 0.05
    grpo:
      # Similar configuration for GRPO
```

## 🔧 Advanced Configuration

### Custom Algorithm Parameters

Each algorithm supports specific parameters:

#### Intuitor Parameters
- Learning rate: `3e-6`
- PPO epochs: `1`
- Warmup style: `cosine`
- GPU memory utilization: `0.8`
- Chunked prefill: `false`

#### GRPO Parameters
- Learning rate: `1e-6`
- PPO epochs: `3`
- Token entropy separation: `true`
- GPU memory utilization: `0.75`
- Chunked prefill: `true`

### Environment Variables

Required environment variables in `.env`:
```bash
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
```

## 📈 Monitoring

### Wandb Metrics

The system logs comprehensive metrics:
- `alternating/current_algorithm`: Current active algorithm
- `alternating/current_phase`: Current phase number
- `alternating/steps_in_phase`: Steps in current phase
- `alternating/current_lr`: Current learning rate
- `alternating/current_ppo_epochs`: Current PPO epochs
- `alternating/using_grpo_normalization`: GRPO-specific flags
- `alternating/using_self_certainty`: Intuitor-specific flags

### Phase History

Each training session maintains detailed phase history:
```json
{
  "phase": 0,
  "algorithm": "intuitor",
  "steps": 50,
  "global_step": 50,
  "timestamp": 1694781234.567
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes in test configuration
2. **KL Loss Errors**: Ensure ref policy is enabled for KL modes
3. **Algorithm Switch Failures**: Check algorithm-specific configurations
4. **Dataset Loading**: Verify file paths and dataset limits

### Debug Mode

Enable detailed logging:
```bash
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
```

## 📝 Examples

### Production Training
```bash
./scripts/train/run_alternating_unified.sh \
    --algorithms "intuitor,grpo" \
    --steps-per-phase 100 \
    --kl-mode kl005 \
    --project-name "Production-Run" \
    --total-epochs 10
```

### Research Experiment
```bash
./scripts/train/run_alternating_unified.sh \
    --algorithms "grpo,intuitor" \
    --start-with grpo \
    --steps-per-phase 25 \
    --kl-mode kl01 \
    --project-name "Research-GRPO-First" \
    --total-epochs 8
```

### Quick Validation
```bash
./scripts/train/run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 100 \
    --project-name "Validation"
```

## 🔗 Related Files

- `scripts/train/run_alternating_unified.sh`: Main training script
- `verl/trainer/config/alternating_official.yaml`: Production config
- `verl/trainer/config/alternating_test.yaml`: Test config
- `verl/trainer/ppo/enhanced_alternating_trainer.py`: Enhanced trainer
- `verl/trainer/main_alternating.py`: Main entry point
