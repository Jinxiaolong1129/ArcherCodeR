# 步骤 1: 获取节点信息（在任一节点执行）
echo $SLURM_JOB_NODELIST  # compute-[718,891]
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "$nodes"  # 应该显示 compute-718 compute-891

# 步骤 2: 获取头节点 IP（在任一节点执行）
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "compute-718" hostname --ip-address)
echo "Head node IP: $head_node_ip"

# 步骤 3: 启动头节点（在 compute-718 执行或通过 srun）
srun --nodes=1 --ntasks=1 -w "compute-718" \
    ray start --head --node-ip-address="$head_node_ip" --port=6379 \
    --num-cpus=64 --num-gpus=4 &

# 步骤 4: 启动工作节点（在 compute-891 执行或通过 srun）
srun --nodes=1 --ntasks=1 -w "compute-891" \
    ray start --address "$head_node_ip:6379" --num-cpus=64 --num-gpus=4 &