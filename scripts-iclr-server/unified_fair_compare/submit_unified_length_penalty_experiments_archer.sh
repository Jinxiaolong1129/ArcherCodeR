#!/usr/bin/env bash
# Submission commands for unified_fair_compare length-penalty experiments.
# Copy one block at a time. One command block = one training job.

# Common length-penalty defaults
# penalty = -alpha * relu(min_len - L) / min_len
# Recommended first run: alpha=1.0, min_len=5000

# ------------------------------
# 1) Intuitor + length penalty (alpha=1.0, min_len=5000)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="lenpen-a1-l5000" \
SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True \
SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 \
SHORT_LEN_PENALTY_ALPHA_OVERRIDE=1.0 \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh


# ------------------------------
# 2) Token Entropy + length penalty (alpha=1.0, min_len=5000)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="lenpen-a1-l5000" \
SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True \
SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 \
SHORT_LEN_PENALTY_ALPHA_OVERRIDE=1.0 \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh


# ------------------------------
# 3) Trajectory Entropy + length penalty (alpha=1.0, min_len=5000)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="lenpen-a1-l5000" \
SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True \
SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 \
SHORT_LEN_PENALTY_ALPHA_OVERRIDE=1.0 \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh


# ------------------------------
# 4) Probability Disparity + length penalty (alpha=1.0, min_len=5000)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="lenpen-a1-l5000" \
SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True \
SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 \
SHORT_LEN_PENALTY_ALPHA_OVERRIDE=1.0 \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh


# ------------------------------
# Optional: Intuitor alpha sweep
# ------------------------------
# alpha=0.5
# EXP_SUFFIX="lenpen-a05-l5000" SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 SHORT_LEN_PENALTY_ALPHA_OVERRIDE=0.5 \
# bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh
#
# alpha=2.0
# EXP_SUFFIX="lenpen-a2-l5000" SHORT_LEN_PENALTY_ENABLE_OVERRIDE=True SHORT_LEN_PENALTY_MIN_LEN_OVERRIDE=5000 SHORT_LEN_PENALTY_ALPHA_OVERRIDE=2.0 \
# bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh
