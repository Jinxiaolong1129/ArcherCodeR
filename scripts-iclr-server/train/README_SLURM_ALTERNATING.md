# SLURM Alternating Training Scripts

基于参考的 Intuitor SLURM 脚本创建的交替训练 SLURM 脚本。

## 📁 文件说明

### 1. `slurm-run_alternating_unified.sh`
完整的交替训练 SLURM 脚本，支持所有参数配置。

**资源配置：**
- 节点数：1
- CPU：180 cores
- 内存：512GB
- GPU：8 张
- 时间限制：48 小时
- 分区：schmidt_sciences

### 2. `slurm-run_alternating_unified-test.sh`
测试模式的 SLURM 脚本，用于快速验证。

**资源配置：**
- 节点数：1
- CPU：180 cores
- 内存：512GB
- GPU：8 张
- 时间限制：12 小时
- 分区：schmidt_sciences

**测试配置：**
- 每个阶段 1 步（快速切换）
- 数据集限制：500 样本
- 总轮数：2
- KL 模式：no-kl

## 🚀 使用方法

### 基本使用

```bash
# 提交默认配置的交替训练任务
sbatch scripts/train/slurm-run_alternating_unified.sh

# 提交测试模式任务
sbatch scripts/train/slurm-run_alternating_unified-test.sh
```

### 自定义参数

```bash
# 自定义算法顺序和步数
sbatch scripts/train/slurm-run_alternating_unified.sh \
    --algorithms "grpo,intuitor" \
    --steps-per-phase 25 \
    --start-with grpo \
    --total-epochs 6

# 使用 KL 损失
sbatch scripts/train/slurm-run_alternating_unified.sh \
    --kl-mode kl005 \
    --project-name "MyProject"

# 测试模式自定义
sbatch scripts/train/slurm-run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 1000 \
    --algorithms "intuitor,grpo"
```

## 📋 支持的参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--algorithms` | 逗号分隔的算法列表 | `intuitor,grpo` |
| `--steps-per-phase` | 每个算法阶段的步数 | `50` |
| `--start-with` | 起始算法 | `intuitor` |
| `--total-epochs` | 总训练轮数 | `4` |
| `--kl-mode` | KL 损失模式 | `no-kl` |
| `--project-name` | Wandb 项目名称 | `ArcherCodeR` |
| `--config` | 配置文件名 | `alternating_official` |
| `--test-mode` | 启用测试模式 | - |
| `--dataset-limit` | 数据集样本限制 | - |

## 🔧 KL 损失模式

- `no-kl`: 无 KL 损失（默认）
- `kl005`: KL 系数 = 0.05
- `kl01`: KL 系数 = 0.1
- `kl02`: KL 系数 = 0.2
- `kl05`: KL 系数 = 0.5

## 📊 输出目录

训练结果将保存在以下目录结构：
```
./output/ArcherCodeR/
├── Alternating-Training/          # 完整训练的 SLURM 日志
├── Alternating-Test/              # 测试模式的 SLURM 日志
└── Alternating-{algorithms}-{kl_mode}-steps{N}-epochs{M}-{timestamp}/  # 训练输出
    ├── eval/                      # 验证数据
    ├── training.log              # 训练日志
    └── global_step_*/            # 检查点
```

## 🔍 监控任务

```bash
# 查看任务状态
squeue -u $USER

# 查看任务详情
scontrol show job <JOB_ID>

# 查看实时日志
tail -f ./output/ArcherCodeR/Alternating-Training/slurm_out_<JOB_ID>.txt

# 查看错误日志
tail -f ./output/ArcherCodeR/Alternating-Training/slurm_error_<JOB_ID>.txt
```

## 🛠️ 故障排除

### Ray 相关问题
脚本会自动清理 Ray 进程和临时文件：
- 停止现有 Ray 集群
- 清理临时文件 (`/tmp/ray`, `/dev/shm/ray*`)
- 设置 Ray CPU 限制为 160

### 环境变量
确保 `.env` 文件包含：
```bash
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
```

### 资源不足
如果遇到资源不足，可以调整 SLURM 参数：
- 减少 `--cpus-per-task`
- 减少 `--mem`
- 调整 `--time` 限制

## 📝 示例命令

```bash
# 1. 基本交替训练（默认配置）
sbatch scripts/train/slurm-run_alternating_unified.sh

# 2. 快速测试
sbatch scripts/train/slurm-run_alternating_unified-test.sh

# 3. 自定义配置
sbatch scripts/train/slurm-run_alternating_unified.sh \
    --algorithms "grpo,intuitor" \
    --steps-per-phase 40 \
    --start-with grpo \
    --kl-mode kl005 \
    --total-epochs 6

# 4. 测试模式自定义
sbatch scripts/train/slurm-run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 1000 \
    --project-name "QuickTest"
```
