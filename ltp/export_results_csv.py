#!/usr/bin/env python3
"""Export ALL eval results into ONE tidy CSV for plotting.

Reads the AUTHORITATIVE scorer outputs (NOT the misleading per-problem .pass.csv for LCB):
  - LiveCodeBench: <dataset>.parquet.pass.lcb.local.csv   cols: pass@1, pass@4, pass@8
  - AIME:          <dataset>.parquet.pass.csv             cols: pass@1, pass@32

Both eval response-length budgets are captured: each checkpoint may have eval_out_8192 AND
eval_out_16384 subdirs (e.g. the 7B swe-rl runs were evaluated at both). The `eval_len` column
distinguishes them -- this is the 8K-vs-16K axis. (Training-length / reward context lives in
`group` / `family`, e.g. r1-1.5b-8k-random = trained at 8K with random reward.)

Output (under ltp/results/):
  results_all.csv   one row per (family, exp, eval_len, step, benchmark) -> ideal for plotting

Unscored cells are omitted (not zero) so the file always reflects what is actually scored;
safe to re-run any time as scoring jobs finish.
"""
import csv, glob, os, re

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEPS = [20, 50, 80, 105]
# benchmark -> scorer kind (decides csv suffix / columns)
BENCHES = [("livecodebench_v5", "lcb"), ("livecodebench_v6", "lcb"),
           ("aime2024", "aime"), ("aime2025", "aime")]

# (family_label, group_dir, model, train_len, [exp_subdirs])
# train_len = RL training max response length (from the training job's MAX_RESP_LEN; default 8192).
# This is independent of eval_len (the eval response budget). e.g. the 7B swe-rl runs are all
# trained at 8K but evaluated at BOTH 8K and 16K.
GROUPS = [
    ("1.5B main",     "r1-1.5b-16k",        "r1-1.5b",  16384, ["grpo", "in", "tok", "traj", "prob"]),
    ("1.5B +KL",      "r1-1.5b-16k-kl",     "r1-1.5b",  16384, ["in-kl", "tok-kl", "traj-kl", "prob-kl"]),
    ("1.5B +LP",      "r1-1.5b-16k-lp",     "r1-1.5b",  16384, ["in-lp", "tok-lp", "traj-lp", "prob-lp"]),
    ("1.5B 16K-rand", "r1-1.5b-16k-random", "r1-1.5b",  16384, ["grpo-random"]),
    ("1.5B 8K-rand",  "r1-1.5b-8k-random",  "r1-1.5b",  8192,  ["grpo-random"]),
    ("4B main",       "qwen3-4b-16k",       "qwen3-4b", 16384, ["grpo", "in", "tok", "traj", "prob"]),
    ("7B main",       "swe-rl",             "coder7b",  8192,  ["swe-rl-grpo", "swe-rl-intuitor", "swe-rl-token_entropy", "swe-rl-trajectory_entropy", "swe-rl-prob_disparity"]),
    ("7B +KL",        "swe-rl-kl",          "coder7b",  8192,  ["swe-rl-intuitor-kl", "swe-rl-token_entropy-kl", "swe-rl-trajectory_entropy-kl", "swe-rl-prob_disparity-kl"]),
    ("7B random",     "swe-rl-random",      "coder7b",  8192,  ["grpo-random"]),
]

# step0 base models (path has no global_step / actor wrapper)
STEP0 = [
    ("1.5B base", "r1-1.5b-base",  "r1-1.5b"),
    ("7B base",   "coder-7b-base", "coder7b"),
]

ABBR = {"in": "intuitor", "tok": "token_entropy", "traj": "trajectory_entropy", "prob": "prob_disparity"}


def parse_method(exp):
    """exp subdir -> (method, variant). e.g. swe-rl-token_entropy-kl -> (token_entropy, kl)."""
    e = exp
    if e.startswith("swe-rl-"):
        e = e[len("swe-rl-"):]
    variant = "none"
    if e.endswith("-kl"):
        variant, e = "kl", e[:-3]
    elif e.endswith("-lp"):
        variant, e = "lenpen", e[:-3]
    if e == "grpo-random":
        return "grpo", "random"
    return ABBR.get(e, e), variant


def read_first_row(path):
    if not os.path.exists(path):
        return None
    try:
        rows = list(csv.DictReader(open(path)))
        return rows[0] if rows else None
    except Exception:
        return None


def values_at(eval_dir, bench, kind):
    """Return {pass@1,pass@4,pass@8,pass@32} for one (eval_dir, benchmark), or None if unscored."""
    suf = ".pass.lcb.local.csv" if kind == "lcb" else ".pass.csv"
    row = read_first_row(os.path.join(eval_dir, f"{bench}.parquet{suf}"))
    if row is None:
        return None
    return {k: row.get(k, "") for k in ("pass@1", "pass@4", "pass@8", "pass@32")}


def eval_len_dirs(parent):
    """Yield (eval_len:int, eval_dir) for every eval_out_<L> subdir present under parent."""
    for d in sorted(glob.glob(os.path.join(parent, "eval_out_*"))):
        m = re.search(r"eval_out_(\d+)$", d)
        if m:
            yield int(m.group(1)), d


os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "results_all.csv")
COLS = ["family", "group", "exp", "model", "method", "variant",
        "train_len", "eval_len", "step", "benchmark", "pass@1", "pass@4", "pass@8", "pass@32"]

n = 0
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    # trained checkpoints
    for family, group, model, train_len, exps in GROUPS:
        for exp in exps:
            method, variant = parse_method(exp)
            for step in STEPS:
                parent = f"{O}/{group}/{exp}/global_step_{step}/actor/hf_model"
                for elen, ed in eval_len_dirs(parent):
                    for bench, kind in BENCHES:
                        v = values_at(ed, bench, kind)
                        if v is None:
                            continue
                        w.writerow([family, group, exp, model, method, variant, train_len, elen, step, bench,
                                    v["pass@1"], v["pass@4"], v["pass@8"], v["pass@32"]])
                        n += 1
    # step0 base models (step = 0, no RL training -> train_len blank)
    for family, name, model in STEP0:
        parent = f"{O}/step0/{name}"
        for elen, ed in eval_len_dirs(parent):
            for bench, kind in BENCHES:
                v = values_at(ed, bench, kind)
                if v is None:
                    continue
                w.writerow([family, "step0", name, model, "base", "none", "", elen, 0, bench,
                            v["pass@1"], v["pass@4"], v["pass@8"], v["pass@32"]])
                n += 1

print(f"wrote {out_path}  ({n} rows)")

# quick coverage summary by (family, eval_len)
cov = {}
for row in csv.DictReader(open(out_path)):
    cov.setdefault((row["family"], row["eval_len"]), 0)
    cov[(row["family"], row["eval_len"])] += 1
print("\ncoverage (rows per family x eval_len):")
for (fam, el), c in sorted(cov.items()):
    print(f"  {fam:14s} eval_len={el:6s} {c:4d} rows")
