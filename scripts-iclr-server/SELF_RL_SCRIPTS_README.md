# Self-RL Training Scripts

这个目录包含了四种Self-RL训练方法的完整训练脚本。

## 📁 目录结构

```
scripts-iclr-server/
├── intuitor/                    # Self-Certainty (INTUITOR)
│   ├── intuitor_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
│   └── slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
├── trajectory_entropy/          # Trajectory-Level Entropy
│   ├── trajectory_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
│   └── slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
├── token_entropy/               # Token-Level Entropy
│   ├── token_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
│   └── slurm-TokenEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
└── prob_disparity/              # Probability Disparity
    ├── prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
    └── slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
```

## 🚀 快速开始

### 方法1: 直接运行（本地或交互式节点）

```bash
# Self-Certainty (INTUITOR)
bash scripts-iclr-server/intuitor/intuitor_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh

# Trajectory-Level Entropy
bash scripts-iclr-server/trajectory_entropy/trajectory_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh

# Token-Level Entropy
bash scripts-iclr-server/token_entropy/token_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh

# Probability Disparity
bash scripts-iclr-server/prob_disparity/prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
```

### 方法2: SLURM提交

```bash
# Self-Certainty (INTUITOR)
sbatch scripts-iclr-server/intuitor/slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh

# Trajectory-Level Entropy
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh

# Token-Level Entropy
sbatch scripts-iclr-server/token_entropy/slurm-TokenEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh

# Probability Disparity
sbatch scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
```

## 📊 方法对比

| 方法 | Advantage Estimator | 公式 | 计算成本 |
|------|-------------------|------|---------|
| **Self-Certainty** | `intuitor` | `1/\|y\| * Σ D_KL(U\|\|π_θ)` | 中等 |
| **Trajectory Entropy** | `trajectory_entropy` | `1/\|y\| * Σ log π_θ(y_t)` | 低（复用log_probs） |
| **Token Entropy** | `token_entropy` | `-1/\|y\| * Σ H(π_θ)` | 中等 |
| **Prob Disparity** | `prob_disparity` | `1/M * Σ [max π - 2nd_max π]` | 中等 |

## ⚙️ 配置说明

所有脚本使用相同的基础配置：

### 模型配置
- **模型**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **最大prompt长度**: 2K tokens
- **最大response长度**: 8K tokens

### 训练配置
- **Batch size**: 64
- **Mini batch size**: 32
- **每个prompt的响应数**: 16
- **训练epochs**: 10
- **学习率**: 3e-6
- **Warmup ratio**: 0.1
- **Weight decay**: 0.1

### 生成配置
- **Temperature**: 1.0
- **Top-p**: 1.0
- **Top-k**: -1 (vLLM模式)
- **Tensor parallel**: 2

### 验证配置
- **验证响应数**: 4
- **验证temperature**: 0.8
- **验证频率**: 每10步

### 资源配置
- **GPU数量**: 8
- **节点数**: 1
- **CPU数**: 180
- **内存**: 512GB
- **时间限制**: 24小时

## 📝 输出位置

训练输出会保存在以下目录：

```
./output/ArcherCodeR/
├── Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/
├── Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/
├── Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/
└── Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/
```

每个目录包含：
- 模型检查点
- 训练日志
- 验证结果（`eval/`子目录）
- SLURM输出（如果使用SLURM）

## 🔍 监控训练

### WandB监控
所有实验会自动上传到WandB项目 `ArcherCodeR`，实验名称如下：
- `Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple`
- `Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple`
- `Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple`
- `Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple`

### 本地日志
训练日志保存在：
```
./output/ArcherCodeR/{exp_name}/{project_name}_{exp_name}_{method}.log
```

### SLURM日志
SLURM输出保存在：
```
./output/ArcherCodeR/{exp_name}/slurm_out_{job_id}.txt
./output/ArcherCodeR/{exp_name}/slurm_error_{job_id}.txt
```

## 🛠️ 自定义配置

如果需要修改配置，可以在命令行添加参数覆盖：

```bash
# 修改batch size
bash scripts-iclr-server/trajectory_entropy/trajectory_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh \
    data.train_batch_size=128

# 修改学习率
bash scripts-iclr-server/token_entropy/token_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh \
    actor_rollout_ref.actor.optim.lr=5e-6

# 修改响应数
bash scripts-iclr-server/prob_disparity/prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh \
    actor_rollout_ref.rollout.n=32
```

## 📋 检查任务状态

### SLURM任务
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务
squeue -j <job_id>

# 取消任务
scancel <job_id>
```

### Ray进程
```bash
# 查看Ray状态
ray status

# 停止Ray
ray stop

# 清理Ray临时文件
rm -rf /tmp/ray /dev/shm/ray*
```

## 🐛 故障排除

### 1. Ray连接问题
如果遇到Ray连接错误，运行：
```bash
ray stop --force
pkill -f ray::
rm -rf /tmp/ray /dev/shm/ray*
```

### 2. GPU内存不足
减小batch size或启用offload：
```bash
bash script.sh \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### 3. 验证失败
如果验证阶段失败，可以禁用初始验证：
```bash
bash script.sh trainer.val_before_train=False
```

## 📚 相关文档

- **实现文档**: `SELF_RL_METHODS_IMPLEMENTATION.md`
- **验证文档**: `IMPLEMENTATION_VERIFICATION.md`
- **主README**: `README.md`

## 🎯 实验建议

### 对比实验设置
1. **相同配置**: 使用相同的超参数运行所有四种方法
2. **相同种子**: 确保数据加载顺序一致
3. **相同硬件**: 在相同的GPU配置上运行
4. **相同数据**: 使用相同的训练和验证数据

### 评估指标
- **Pass@1**: 单次采样通过率
- **Pass@k**: k次采样通过率
- **训练稳定性**: Loss曲线平滑度
- **收敛速度**: 达到目标性能的步数
- **计算效率**: 每步训练时间

### 分析维度
1. **性能对比**: 哪种方法最终性能最好？
2. **效率对比**: 哪种方法训练最快？
3. **稳定性对比**: 哪种方法最稳定？
4. **任务依赖**: 不同方法在不同任务上的表现

## ✅ 运行检查清单

在运行实验前，确保：
- [ ] `.env`文件存在且包含`WANDB_API_KEY`和`HF_TOKEN`
- [ ] 数据文件存在于`./data/`目录
- [ ] 有足够的磁盘空间保存检查点
- [ ] GPU资源可用（8张GPU）
- [ ] Ray环境已清理（如果之前运行过）

## 🎉 开始实验

现在你可以开始运行实验了！建议先运行一个方法验证配置正确，然后并行运行所有四种方法进行对比。

祝实验顺利！🚀

