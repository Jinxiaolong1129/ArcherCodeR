#!/usr/bin/env bash
# 修改为你的实际checkpoint路径
model_path=./output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/global_step_90/actor

/data/xuandong_zhao/anaconda3/envs/archer/bin/python -m tools.model_merge merge \
    --backend fsdp \
    --local_dir ${model_path} \
    --target_dir ${model_path}/hf_model