#!/bin/bash
#
# Submit one 8-GPU node and run 4+4 parallel eval:
#   - Worker A: step 50 on GPU 0,1,2,3
#   - Worker B: step 80 on GPU 4,5,6,7
#

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --job-name=aime24-c10-4p4
#SBATCH --output=/data_storage/wyj/jxl/ArcherCodeR/Archer_eval/tools/logs/slurm_aime24_c10_4p4_%j.out
#SBATCH --error=/data_storage/wyj/jxl/ArcherCodeR/Archer_eval/tools/logs/slurm_aime24_c10_4p4_%j.err

set -euo pipefail

PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
EVAL_DIR="${PROJ_DIR}/Archer_eval"
mkdir -p "${EVAL_DIR}/tools/logs"

# Optional proxy (uncomment if needed)
# export HTTP_PROXY="http://100.68.168.184:3128"
# export HTTPS_PROXY="http://100.68.168.184:3128"

# If you do NOT want merge in this job, set MERGE_IF_NEEDED=0 before sbatch.
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}"

cd "${EVAL_DIR}"
export PYTHONPATH="${EVAL_DIR}:${PYTHONPATH:-}"

# Avoid stale ray from previous jobs
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "MERGE_IF_NEEDED=${MERGE_IF_NEEDED}"
echo "Start worker A(step50) + worker B(step80)"

# Worker A: step 50 (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
STEP_LIST=50 \
MERGE_IF_NEEDED="${MERGE_IF_NEEDED}" \
bash tools/eval_aime24_clip_c10_steps50_80.sh \
  > "tools/logs/aime24_c10_step50_job${SLURM_JOB_ID}.log" 2>&1 &
PID_A=$!

# Worker B: step 80 (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS=4 \
STEP_LIST=80 \
MERGE_IF_NEEDED="${MERGE_IF_NEEDED}" \
bash tools/eval_aime24_clip_c10_steps50_80.sh \
  > "tools/logs/aime24_c10_step80_job${SLURM_JOB_ID}.log" 2>&1 &
PID_B=$!

wait ${PID_A}
STATUS_A=$?
wait ${PID_B}
STATUS_B=$?

echo "workerA(step50) exit=${STATUS_A}"
echo "workerB(step80) exit=${STATUS_B}"

if [[ ${STATUS_A} -ne 0 || ${STATUS_B} -ne 0 ]]; then
  echo "At least one worker failed."
  exit 1
fi

echo "All workers finished successfully."

