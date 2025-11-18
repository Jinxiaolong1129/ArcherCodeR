# 查看队列中的任务
squeue -p schmidt_sciences

# 查看分区节点状态
sinfo -p schmidt_sciences

# sinfo 输出说明：
# PARTITION: 分区名称
# AVAIL: 可用性 (up=可用, down=不可用)
# TIMELIMIT: 时间限制
# NODES: 节点数量
# STATE: 节点状态
#   - idle: 空闲，可以接受新任务
#   - mix: 混合使用，部分资源被占用，还有空闲资源
#   - alloc: 完全分配，所有资源都被占用
#   - drng (draining): 正在排空，不接受新任务，等待当前任务完成
#   - down: 节点关闭
# NODELIST: 节点列表


# 提交第一个作业
sbatch scripts/train/slurm-run_Archer-Qwen2.5-1.5B-2K-8K-16resp.sh

# 提交第二个作业  
sbatch scripts/train/slurm-run_Archer-Qwen2.5-1.5B-2K-16K-16resp.sh

# 提交第三个作业
sbatch scripts/train/slurm-run_Archer-Qwen2.5-3B-2K-8K-16resp.sh




# 查看自己所有任务的详细信息
for job in $(squeue -u xuandong_zhao -h -o "%A"); do
  echo "========== JOB $job =========="
  scontrol show job $job
done

# 查看排队任务需要的GPU数量
squeue -p schmidt_sciences -t PD -o "%.10i %.15j %.12u %.8T %.10M %.6D %.20b %.20R"

# 查看排队任务的详细GPU信息
for job in $(squeue -p schmidt_sciences -t PD -h -o "%A"); do
  echo "========== JOB $job =========="
  scontrol show job $job | grep -E "JobId|JobName|UserId|Gres|TresPerNode"
done

# 查看所有任务（包括运行中和排队中）的GPU信息
squeue -p schmidt_sciences -o "%.10i %.15j %.12u %.8T %.10M %.6D %.20b %.20R"
