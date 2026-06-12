#!/usr/bin/env bash
# N sweep submission commands.
# Copy one block at a time. One command block = one training job.

# ---------- Intuitor | n=8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n8" N_RESP_OVERRIDE="8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh

# ---------- Intuitor | n=12 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n12" N_RESP_OVERRIDE="12" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh

# ---------- Pure-GRPO | n=8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n8" N_RESP_OVERRIDE="8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Pure-GRPO | n=12 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n12" N_RESP_OVERRIDE="12" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Trajectory Entropy | n=8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n8" N_RESP_OVERRIDE="8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Trajectory Entropy | n=12 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n12" N_RESP_OVERRIDE="12" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Token Entropy | n=8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n8" N_RESP_OVERRIDE="8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Token Entropy | n=12 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n12" N_RESP_OVERRIDE="12" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Prob Disparity | n=8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n8" N_RESP_OVERRIDE="8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh

# ---------- Prob Disparity | n=12 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="n12" N_RESP_OVERRIDE="12" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh
