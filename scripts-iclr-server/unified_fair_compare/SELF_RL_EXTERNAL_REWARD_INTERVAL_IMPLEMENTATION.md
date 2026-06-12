# selfrl_external_reward_interval 实施说明

## 1. 目标

为 self-RL 方法增加 **log-only 外部 reward 监控**，并通过 `selfrl_external_reward_interval` 控制计算频率：

- 不改变 `adv_estimator`（仍使用 intrinsic 信号训练）
- 不改变 `advantages/returns` 计算路径
- 仅把外部 reward 统计打到 W&B（或当前 tracklogger 后端）

---

## 2. 配置项定义

配置键：

- `trainer.selfrl_external_reward_interval`

语义：

- 每隔 `k` 个 global step 计算一次外部 reward 监控
- 例如 `k=5` 表示 step 5, 10, 15, ... 触发

默认值：

- `5`（未配置时自动使用）

关闭方式：

- 设为 `0` 或负数，关闭该功能

---

## 3. 生效范围

仅对以下 self-RL 算法生效：

- `intuitor`
- `intuitor_entropy`
- `token_entropy`
- `trajectory_entropy`
- `prob_disparity`

不对以下算法生效：

- `grpo`（本身已使用外部 reward 训练）
- `intuitor_selective`（其训练逻辑不同）
- 其他非上述估计器

---

## 4. 实现位置

核心修改文件：

- `verl/trainer/ppo/ray_trainer.py`

新增/改动点：

1. `RayPPOTrainer._should_log_external_reward()`
   - 判断是否满足：
     - 当前算法属于 self-RL 生效列表
     - `val_reward_fn` 可用
     - `trainer.selfrl_external_reward_interval > 0`
     - `global_steps % interval == 0`

2. `RayPPOTrainer._compute_external_reward_log_metrics(batch)`
   - 调用 `self.val_reward_fn(batch, return_dict=True)` 得到 `reward_tensor`
   - 计算序列级分数：`seq_reward = reward_tensor.sum(-1)`
   - 产出统计指标并返回 dict

3. 训练主循环插入点
   - 在常规 reward 计算后、更新阶段前做 log-only 计算
   - 指标写入 `metrics.update(...)`
   - 不写回 `token_level_scores` / `token_level_rewards`，不影响优化

---

## 5. 记录到监控系统的指标

每次触发时记录：

- `external_reward/mean`
- `external_reward/max`
- `external_reward/min`
- `external_reward/std`
- `external_reward/positive_ratio`
- `external_reward/num_samples`

按数据源（若 batch 中含 `data_source`）额外记录：

- `external_reward/by_data_source/<source>/mean`
- `external_reward/by_data_source/<source>/positive_ratio`
- `external_reward/by_data_source/<source>/num_samples`

异常保护：

- 计算失败不会中断训练
- 会写入 `external_reward/log_error=1.0` 并打印 warning

---

## 6. 与训练行为的关系（重要）

该功能是 **纯监控**：

- 不参与 `compute_advantage(...)`
- 不覆盖 `token_level_scores` / `token_level_rewards`
- 不改变 actor/critic 更新结果

因此可以把它理解为“训练中在线评测采样”，而不是优化目标的一部分。

---

## 7. 如何配置（命令行/脚本）

在训练命令追加（示例）：

```bash
+trainer.selfrl_external_reward_interval=5
```

改为每 10 step：

```bash
+trainer.selfrl_external_reward_interval=10
```

关闭：

```bash
+trainer.selfrl_external_reward_interval=0
```

---

## 8. 开销与建议

开销来源：

- 每次触发会额外执行一次外部 reward 计算（对 code 任务可能较重）

建议：

- 默认 `k=5` 适合先观察趋势
- 若训练变慢明显，增大到 `k=10` 或 `k=20`
- 若只看粗粒度趋势，可在中后期降低频率

---

## 9. 验收检查

可用以下方式确认功能生效：

1. 日志中 step 到达 `k` 的倍数时，出现 `external_reward/*` 指标
2. W&B 面板可检索到 `external_reward/mean`
3. self-RL 原有 intrinsic 指标仍正常变化（如 `intuitor/*`、`internal_metrics/*`）
4. 训练 loss/adv 曲线行为与未接入前趋势一致（仅多了监控开销）

---

## 10. 常见问题

1) 为什么不是每步都算？  
因为外部 reward 计算昂贵，间隔触发更实用。

2) 为什么复用 `val_reward_fn`？  
self-RL 训练路径下 `reward_fn` 往往是 dummy；`val_reward_fn` 才是实际外部评测函数。

3) 是否支持按 prompt 展开 rollout 明细？  
当前实现是标量聚合指标；若需要可后续扩展为 W&B Table / 本地 JSONL 记录。

