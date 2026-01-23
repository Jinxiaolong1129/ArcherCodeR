#!/bin/bash
#SBATCH --job-name=eval-certainty
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval-certainty-%j.log
#SBATCH --error=logs/eval-certainty-%j.err

# ============ Usage ============
# sbatch slurm-eval_with_certainty.sh <model_path> <dataset> <training_method>
#
# Examples:
# sbatch slurm-eval_with_certainty.sh ./output/Archer-Intuitor-Qwen2.5-1.5B/global_step_100/actor livecodebench_v5 INTUITOR
# sbatch slurm-eval_with_certainty.sh ./output/Archer-TrajectoryEntropy-Qwen2.5-1.5B/global_step_100/actor livecodebench_v5 TRAJECTORY_ENTROPY

set -x

# Get arguments
MODEL_PATH=${1:-"./model/WizardCodeR-1.5B-DAPO"}
DATASET=${2:-"livecodebench_v5"}
TRAINING_METHOD=${3:-"INTUITOR"}

# Create logs directory
mkdir -p logs

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../../..

# Run the evaluation script
bash scripts-iclr-server/eval/run_eval_with_certainty.sh ${MODEL_PATH} ${DATASET} ${TRAINING_METHOD}

