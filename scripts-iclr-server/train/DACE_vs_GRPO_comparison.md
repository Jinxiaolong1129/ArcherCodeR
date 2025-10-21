# DACE vs GRPO 配置对比

## 📊 DACE默认参数

根据**DACE论文**和代码定义（`verl/trainer/ppo/core_algos.py` 第312-320行）：

> 论文原文: "For DACE, our default configuration uses a scaling factor of **αscale = 0.05** and a difficulty threshold of **βthreshold = 0.4**."

```python
@register_adv_est(AdvantageEstimator.DACE)
def compute_dace_advantage(
    token_level_rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    alpha_scale: float = 0.05,             # ⭐ 论文默认值
    beta_threshold: float = 0.4,           # ⭐ 论文默认值
    epsilon: float = 1e-6,                 # ⭐ 默认值
    norm_adv_by_std_in_grpo: bool = True,  # ⭐ 默认值
    config=None,
    **kwargs,
):
```

**DACE参数说明**：

| 参数 | 论文默认值 | 含义 | 取值范围 |
|------|-----------|------|----------|
| `alpha_scale` | **0.05** | 内在奖励缩放因子 | 0.03-0.2 推荐 |
| `beta_threshold` | **0.4** | 难度阈值 | 0.3-0.7 推荐 |
| `epsilon` | 1e-6 | 数值稳定性常数 | 固定 |
| `norm_adv_by_std_in_grpo` | True | 是否用标准差归一化 | True/False |

---

## 🔍 脚本对比

### 共同点 ✅

| 配置项 | DACE | GRPO | 说明 |
|-------|------|------|------|
| **模型路径** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 相同 | ✅ 使用同一个模型 |
| **数据文件** | archercoder-1.5b-train.json | 相同 | ✅ 训练数据相同 |
| **测试文件** | livecodebench_v5.json | 相同 | ✅ 验证数据相同 |
| **max_prompt_length** | 2048 (2K) | 相同 | ✅ |
| **max_response_length** | 8192 (8K) | 相同 | ✅ |
| **train_prompt_bsz** | 64 | 相同 | ✅ |
| **train_prompt_mini_bsz** | 32 | 相同 | ✅ |
| **n_resp_per_prompt** | 16 | 相同 | ✅ 重要！DACE需要多响应 |
| **temperature** | 1.0 | 相同 | ✅ |
| **learning_rate** | 1e-6 | 相同 | ✅ |
| **ppo_epochs** | 3 | 相同 | ✅ |
| **总训练epochs** | 10 | 相同 | ✅ |
| **KL设置** | 全部False/0.0 | 相同 | ✅ 都不使用KL |
| **tensor_parallel** | 2 | 相同 | ✅ |

---

### 核心差异 ⚠️

#### 1. **算法选择（最关键！）**

```bash
# DACE脚本
adv_estimator=dace                        # ⭐ 使用DACE算法

# GRPO脚本
adv_estimator=grpo                        # 使用GRPO算法
```

#### 2. **DACE特有参数（新增）**

```bash
# DACE脚本中新增的配置
algorithm.dace_alpha_scale=0.1            # ⭐ 内在奖励强度
algorithm.dace_beta_threshold=0.5         # ⭐ 难度阈值
algorithm.norm_adv_by_std_in_grpo=True    # 归一化方式
```

**GRPO脚本中不存在这些参数！**

#### 3. **Token Entropy Separate（不同）**

```bash
# DACE脚本
use_token_entropy_separate=False          # ⭐ 禁用（使用纯DACE）

# GRPO脚本
use_token_entropy_separate=True           # 启用（使用token entropy masking）
```

这是一个重要差异！解释：
- **GRPO**: 使用token-level的entropy masking来区分不同确定性的token
- **DACE**: 不需要这个，因为DACE有自己的certainty-based机制

#### 4. **项目和实验名称**

```bash
# DACE脚本
project_name='ArcherCodeR-DACE'
exp_name='DACE-Qwen2.5-1.5B-2K-8K-16resp'

# GRPO脚本
project_name='ArcherCodeR'
exp_name='Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl'
```

---

## 📋 完整参数差异表

### 新增参数（DACE独有）

| 参数 | DACE值 | GRPO | 说明 |
|------|--------|------|------|
| `algorithm.adv_estimator` | **dace** | grpo | ⭐ 核心差异 |
| `algorithm.dace_alpha_scale` | **0.05** | - | ⭐ DACE特有（论文默认） |
| `algorithm.dace_beta_threshold` | **0.4** | - | ⭐ DACE特有（论文默认） |
| `algorithm.norm_adv_by_std_in_grpo` | **True** | (默认True) | 明确指定 |

### 修改参数

| 参数 | DACE值 | GRPO值 | 原因 |
|------|--------|--------|------|
| `use_token_entropy_separate` | **False** | **True** | DACE不需要 |
| `project_name` | ArcherCodeR-DACE | ArcherCodeR | 区分项目 |
| `exp_name` | DACE-Qwen2.5-1.5B-2K-8K-16resp | Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl | 实验命名 |

### Token Entropy相关参数（在DACE中不起作用）

由于 `use_token_entropy_separate=False`，以下参数在DACE中被禁用：

```bash
token_entropy_quantile=0.8
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.5
high_entropy_clip_ratio_high=0.5
```

这些参数在DACE脚本中仍然存在，但因为 `use_token_entropy_separate=False`，它们不会被使用。

---

## 🎯 配置建议

### 基础配置（从GRPO迁移到DACE）

如果您已有GRPO脚本，最小改动：

```bash
# 1. 修改算法
adv_estimator=dace

# 2. 添加DACE参数（可选，使用论文默认值）
algorithm.dace_alpha_scale=0.05      # 论文默认
algorithm.dace_beta_threshold=0.4    # 论文默认
algorithm.norm_adv_by_std_in_grpo=True

# 3. 禁用token entropy（推荐）
use_token_entropy_separate=False
```

### DACE超参数调优

#### alpha_scale（内在奖励强度）

```bash
# 保守设置（训练不稳定时）
algorithm.dace_alpha_scale=0.03

# 标准设置（推荐，论文默认）
algorithm.dace_alpha_scale=0.05

# 增强设置（需要更强信号）
algorithm.dace_alpha_scale=0.1

# 激进设置（实验性）
algorithm.dace_alpha_scale=0.15
```

**选择指南**：
- **首次使用 → 用0.05（论文默认）**
- 训练不稳定 → 减小到0.03
- 需要更强探索/利用 → 增大到0.1
- 数据集困难且模型探索不足 → 最多0.15

#### beta_threshold（难度阈值）

```bash
# 偏简单（更多exploitation）
algorithm.dace_beta_threshold=0.3

# 平衡（推荐，论文默认）
algorithm.dace_beta_threshold=0.4

# 中等
algorithm.dace_beta_threshold=0.5

# 偏困难（更多exploration）
algorithm.dace_beta_threshold=0.6
```

**选择指南**：
- **首次使用 → 用0.4（论文默认）**
- 数据集整体简单（如基础算术）→ 用0.3
- 数据集难度均衡 → 用0.5
- 数据集整体困难（如竞赛题）→ 用0.6-0.7

---

## 🧪 实验对比建议

### 控制变量对比

为了公平对比DACE和GRPO，建议：

#### 选项1: 严格控制（推荐）

```bash
# 两个脚本都设置
use_token_entropy_separate=False    # 都不用token entropy
# 其他参数保持完全一致
# 只改变 adv_estimator
```

#### 选项2: 各用最佳配置

```bash
# GRPO: 用token entropy（GRPO的优化）
use_token_entropy_separate=True

# DACE: 不用token entropy（DACE有自己的机制）
use_token_entropy_separate=False
```

### 对比指标

| 维度 | 指标 | 说明 |
|------|------|------|
| **性能** | Pass@1, Pass@4 | 最终准确率 |
| **效率** | 收敛速度 | 达到目标准确率的steps |
| **稳定性** | Loss曲线 | 训练过程波动 |
| **泛化** | 验证集性能 | 避免过拟合 |

---

## 📝 完整参数清单

### 必须指定的DACE参数

```bash
algorithm.adv_estimator=dace              # 必须！
```

### 推荐指定的DACE参数

```bash
algorithm.dace_alpha_scale=0.05           # 推荐明确指定（论文默认）
algorithm.dace_beta_threshold=0.4         # 推荐明确指定（论文默认）
algorithm.norm_adv_by_std_in_grpo=True    # 推荐明确指定
```

### 推荐修改的其他参数

```bash
use_token_entropy_separate=False          # 推荐禁用
project_name='ArcherCodeR-DACE'           # 建议修改（区分实验）
exp_name='DACE-...'                       # 建议修改（区分实验）
```

---

## ⚙️ 快速配置检查

### DACE脚本检查清单

- [ ] `adv_estimator=dace` ✅
- [ ] `dace_alpha_scale` 已设置（或使用默认0.1）
- [ ] `dace_beta_threshold` 已设置（或使用默认0.5）
- [ ] `n_resp_per_prompt >= 8`（DACE需要多响应估计difficulty）
- [ ] `use_token_entropy_separate=False`（推荐）
- [ ] 其他参数与baseline一致（除非有特殊原因）

### GRPO脚本检查清单

- [ ] `adv_estimator=grpo` ✅
- [ ] `n_resp_per_prompt >= 4`（GRPO基本要求）
- [ ] `use_token_entropy_separate` 根据需要设置
- [ ] 无DACE相关参数

---

## 🎓 总结

### 关键发现

1. **核心差异只有3个参数**:
   - `adv_estimator=dace` (vs `grpo`)
   - `dace_alpha_scale=0.05` (新增，论文默认)
   - `dace_beta_threshold=0.4` (新增，论文默认)

2. **其他配置基本相同**:
   - 模型、数据、批次大小等都一样
   - 训练超参数完全相同

3. **Token Entropy的差异**:
   - DACE禁用 (`False`)
   - GRPO启用 (`True`)
   - 这反映了两种不同的探索机制

### DACE的优势

1. ✅ **自适应**: 根据任务难度自动调整策略
2. ✅ **统一**: 简单和困难任务用同一套参数
3. ✅ **高效**: 简单任务快速收敛，困难任务持续探索
4. ✅ **兼容**: 可以平滑从GRPO切换到DACE

### 迁移建议

从GRPO迁移到DACE：

```bash
# 最小改动（3行，使用论文默认值）
adv_estimator=dace
algorithm.dace_alpha_scale=0.05      # 论文默认
algorithm.dace_beta_threshold=0.4    # 论文默认

# 推荐改动（+1行）
use_token_entropy_separate=False

# 完成！其他参数保持不变
```

