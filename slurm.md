squeue -p schmidt_sciences

sinfo -p schmidt_sciences


# 提交第一个作业
sbatch scripts/train/slurm-run_Archer-Qwen2.5-1.5B-2K-8K-16resp.sh

# 提交第二个作业  
sbatch scripts/train/slurm-run_Archer-Qwen2.5-1.5B-2K-16K-16resp.sh

# 提交第三个作业
sbatch scripts/train/slurm-run_Archer-Qwen2.5-3B-2K-8K-16resp.sh