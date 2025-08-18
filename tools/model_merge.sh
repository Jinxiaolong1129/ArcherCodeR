#!/usr/bin/env bash
# 修改为你的实际checkpoint路径
model_path=./output/ArcherCodeR/Archer-Qwen2.5-3B-2K-8K-16resp/global_step_80/actor

/data/xuandong_zhao/anaconda3/envs/archer/bin/python -m tools.model_merge merge \
    --backend fsdp \
    --local_dir ${model_path} \
    --target_dir ${model_path}/hf_model