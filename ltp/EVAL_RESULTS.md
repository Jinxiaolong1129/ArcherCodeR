# Eval Results — how to extract the table data

How the internal-RL comparison results are stored, which files are authoritative, and how to
turn them into ONE tidy CSV for plotting / the paper table.

All paths are on **cephfs**, reachable only from the remote (`ssh ltp-debug-cpu-2`):
`/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output/`

---

## 1. Directory layout

```
output/
  <group>/<exp>/global_step_<STEP>/actor/hf_model/eval_out_<EVAL_LEN>/
      <benchmark>.parquet                          # generation (GPU) — the model's outputs
      <benchmark>.parquet.pass.lcb.local.csv       # LCB score  (authoritative)
      <benchmark>.parquet.pass.csv                 # AIME score (authoritative) / LCB per-problem summary (NOT for LCB pass@1)
  step0/<base-name>/eval_out_<EVAL_LEN>/           # step0 = before RL (base model), no global_step wrapper
```

- **STEP** ∈ {20, 50, 80, 105}. 105 = final checkpoint (the value used in the paper table).
- **EVAL_LEN** ∈ {8192, 16384} = the eval **response-length budget**. A single checkpoint can have
  BOTH `eval_out_8192/` and `eval_out_16384/` — see the 8K-vs-16K note below.
- **benchmark** ∈ {`livecodebench_v5`, `livecodebench_v6`, `aime2024`, `aime2025`}.
- One eval run produces all 4 benchmarks for a given (exp, step, eval_len).

### groups / exps

| group dir | model | exps (subdirs) | eval_len present |
|---|---|---|---|
| `r1-1.5b-16k` | DeepSeek-R1-Distill-Qwen-1.5B (16K) | grpo, in, tok, traj, prob | 16K |
| `r1-1.5b-16k-kl` | 〃 | in-kl, tok-kl, traj-kl, prob-kl | 16K |
| `r1-1.5b-16k-lp` | 〃 (length penalty) | in-lp, tok-lp, traj-lp, prob-lp | 16K |
| `r1-1.5b-16k-random` | 〃 (random reward) | grpo-random | 16K |
| `r1-1.5b-8k-random` | 〃 (trained at 8K, random reward) | grpo-random | **8K + 16K** |
| `qwen3-4b-16k` | Qwen3-4B | grpo, in, tok, traj, prob | 16K |
| `swe-rl` | Qwen2.5-Coder-7B-Instruct | swe-rl-{grpo,intuitor,token_entropy,trajectory_entropy,prob_disparity} | **8K + 16K** |
| `swe-rl-kl` | 〃 | swe-rl-{intuitor,token_entropy,trajectory_entropy,prob_disparity}-kl | **8K + 16K** |
| `swe-rl-random` | 〃 | grpo-random | **8K + 16K** |
| `step0/{r1-1.5b-base, coder-7b-base}` | base (before RL) | — | **8K + 16K** |

method shorthands: `in`=intuitor, `tok`=token_entropy, `traj`=trajectory_entropy, `prob`=prob_disparity.

> **8K vs 16K.** Only the **7B swe-rl** family, the **1.5B 8K-random** run, and the **step0 base**
> models were evaluated at *both* 8K and 16K. The 1.5B (main/+KL/+LP) and the 4B families were
> evaluated at **16K only**. **`r1-1.5b-12k` was trained but never evaluated** (no eval dirs exist),
> so 12K does not appear in the results.

---

## 2. Which file is authoritative (IMPORTANT)

LCB and AIME use **different scorers** that write **different files**. Read the right one:

| benchmark | authoritative file | pass@1 column lives in |
|---|---|---|
| `livecodebench_v5` / `v6` | `<bench>.parquet.pass.lcb.local.csv` | `pass@1` (also `pass@4`, `pass@8`) |
| `aime2024` / `aime2025` | `<bench>.parquet.pass.csv` | `pass@1` (also `pass@32`) |

⚠️ **For LCB, do NOT read `.pass.csv`** — that is a per-problem summary and reading its aggregate
as pass@1 is what produced the bogus "LCB-v6 all 0" reading earlier. The real LCB pass@1 is only
in `.pass.lcb.local.csv`. (For AIME, `.pass.csv` *is* the right file.)

CSV headers:
```
LCB  .pass.lcb.local.csv : parquet_file,testcase_file,dataset,pass@1,pass@4,pass@8
AIME .pass.csv           : model_path,dataset,pass@1,pass@32
```

---

## 3. Extract → ONE CSV

```bash
ssh ltp-debug-cpu-2 'cd /mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR && \
  python3 ltp/export_results_csv.py'
```

Writes a single consolidated file: **`ltp/results/results_all.csv`** — tidy/long, one row per
(family, exp, train_len, eval_len, step, benchmark):

```
family,group,exp,model,method,variant,train_len,eval_len,step,benchmark,pass@1,pass@4,pass@8,pass@32
```

- `method` ∈ {grpo, intuitor, token_entropy, trajectory_entropy, prob_disparity, base}
- `variant` ∈ {none, kl, lenpen, random}
- `train_len` = RL **training** max response length: 8192 for all 7B swe-rl + 1.5B-8k-rand,
  16384 for 1.5B (main/+KL/+LP/16k-rand) + 4B, blank for step0 base. (1.5B-12k was trained but
  never evaluated, so it does not appear.)
- `eval_len` ∈ {8192, 16384} — **the 8K-vs-16K eval axis** (eval response budget). Independent of
  `train_len`: the 7B runs are train_len=8192 but were evaluated at both eval_len 8192 and 16384.
- `step` ∈ {0, 20, 50, 80, 105}. step 0 = base model (before RL).
- `pass@4`/`pass@8` filled for LCB only; `pass@32` for AIME only; `pass@1` for both.
- Unscored cells are **omitted** (not zero) — the file always reflects what is actually scored.

The script auto-discovers every `eval_out_<L>` dir present, so 8K and 16K are both captured
without any flag. Re-run any time scoring jobs finish to refresh.

---

## 4. Plotting hints (results_all.csv)

```python
import pandas as pd
df = pd.read_csv("ltp/results/results_all.csv")

# 8K vs 16K, final step, LCB v5 — same models, two eval budgets
cmp = df[(df.family == "7B main") & (df.step == 105) & (df.benchmark == "livecodebench_v5")]
cmp.pivot_table(index="method", columns="eval_len", values="pass@1")

# training curve: pass@1 vs step, one line per method, fixed eval_len + benchmark
sub = df[(df.family == "1.5B main") & (df.eval_len == 16384) & (df.benchmark == "livecodebench_v6")]
sub.pivot_table(index="step", columns="method", values="pass@1").plot()

# paper table (final-step pass@1, 16K): one-line pivot, no second CSV needed
final = df[(df.step == 105) & (df.eval_len == 16384)]
final.pivot_table(index=["family", "method", "variant"], columns="benchmark", values="pass@1")
```

`variant` separates the main run (`none`) from ablations (`kl`, `lenpen`, `random`) within the
same `method`/`model`. `eval_len` separates 8K from 16K.

---

## 5. Audit coverage (which step/benchmark cells are scored)

```bash
ssh ltp-debug-cpu-2 'cd /mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR && \
  python3 ltp/audit_steps.py'
```
Legend per cell, in step order 20/50/80/105: `✓`=scored, `g`=parquet generated but not scored, `.`=parquet missing.
(NOTE: `audit_steps.py` currently checks `eval_out_16384` only.)

---

## 6. Re-scoring (CPU only, no GPU)

Generation (GPU) for all experiments is complete; only LCB scoring may have gaps. Scoring is a
pure-CPU job on the `cpu2` VC. Render the template and submit:

```bash
TS=$(date +%Y%m%d-%H%M%S); NAME="score-xxx-$TS"
sed -e "s/__NAME__/$NAME/g" -e "s/__GROUP__/<group>/g" \
    -e "s#__BENCHES__#livecodebench_v6:v6:8#g" \
    ltp/eval-score-cpu.job.yaml.template > ltp/_rendered_$NAME.yaml
bash /mnt/cephfs/data/processing/xiaolong.jin/code/LTP_script/submit_job.sh ltp/_rendered_$NAME.yaml
```
- `__GROUP__` matches the group cases in `ltp/run_eval_group_ltp.sh` (e.g. `4b`, `7b-new-16k`,
  `7b-new-8k`, `1.5b-16krand`, `1.5b-8krand-8k`).
- `__BENCHES__` format: `dataset:lcb_version:n_samples`, space-separated. Limit to what's missing
  (e.g. `livecodebench_v6:v6:8`) — a skip-guard skips already-scored cells, so re-running is safe.
