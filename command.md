squeue -p schmidt_sciences

sinfo -p schmidt_sciences

# 查看分区内所有节点概览
sinfo -p schmidt_sciences -l


# 查看单个节点详细信息
scontrol show node compute-556

# 查看多个节点
scontrol show node compute-556,compute-718,compute-837,compute-891,compute-967

# 或者查看整个分区的所有节点
scontrol show partition schmidt_sciences


# 显示节点名、状态、CPU、内存、GPU等信息
sinfo -p schmidt_sciences -o "%.15N %.6t %.14C %.8m %.10G %.15f"

# 显示节点的负载信息
sinfo -p schmidt_sciences -o "%.15N %.6t %.14C %.8O %.8m"


scontrol show node compute-556,compute-837 | grep -E "(NodeName|CPUTot|Sockets|CoresPerSocket|ThreadsPerCore)"