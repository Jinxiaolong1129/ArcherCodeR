# 查看队列中的任务
squeue -p schmidt_sciences

# 查看分区节点状态
sinfo -p schmidt_sciences

```bash
# 1. Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh

# 2. Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh

# 3. Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2
sbatch scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh

# 4. Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12
sbatch scripts-iclr-server/intuitor/slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh

# 5. Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl
sbatch scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl.sh
```

## 一键提交所有任务

```bash
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh && \
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh && \
sbatch scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh && \
sbatch scripts-iclr-server/intuitor/slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh && \
sbatch scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl.sh
```

## 脚本对应表

| # | 实验名称 | Slurm 脚本相对路径 |
|---|---------|-------------------|
| 1 | Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12 | `scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh` |
| 2 | Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2 | `scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh` |
| 3 | Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2 | `scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-temp1.2.sh` |
| 4 | Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12 | `scripts-iclr-server/intuitor/slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-n12.sh` |
| 5 | Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl | `scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl.sh` |


sbatch scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp0.8.sh && \
sbatch scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp1.2.sh && \
sbatch scripts-iclr-server/train/slurm-Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl.sh
