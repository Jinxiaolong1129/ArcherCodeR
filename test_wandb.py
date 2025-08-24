import wandb

api = wandb.Api()

# 如果你知道run ID，可以直接访问
# run_id = "your-actual-run-id"  # 从wandb界面URL中获取
# target_run = api.run(f"jxl-dragon/ArcherCodeR/{run_id}")

# 或者先找到run ID
runs = api.runs("jxl-dragon/ArcherCodeR")
for run in runs:
    if run.name == "Archer-Qwen2.5-1.5B-2K-16K-16resp":
        run_id = run.id
        print(f"找到run ID: {run_id}")
        
        # 现在可以直接通过ID访问
        target_run = api.run(f"jxl-dragon/ArcherCodeR/{run_id}")
        
        # 获取指标
        metrics = list(target_run.summary.keys())
        print("所有指标名称:")
        for metric in metrics:
            print(f"  {metric}")
        break