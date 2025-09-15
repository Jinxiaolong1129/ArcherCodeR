import wandb

wandb.login(key="5c271ef60b4c4753def92be733cf80487f0c7e78")
api = wandb.Api()

# 使用你的具体 run 路径
run = api.run("jxl-dragon/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp")

# 方法 1: 获取 summary 中的 metrics（最终值）
print("=== Summary Metrics (最终值) ===")
summary_metrics = list(run.summary.keys())
for i, name in enumerate(summary_metrics, 1):
    print(f"{i:2d}. {name}")

print(f"\n总共 {len(summary_metrics)} 个 summary metrics")