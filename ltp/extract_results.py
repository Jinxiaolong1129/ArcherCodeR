#!/usr/bin/env python3
# Extract step-105 pass@1 across all ArcherCodeR eval experiments (eval_out_16384).
import csv, os

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"
DS = ["livecodebench_v5", "livecodebench_v6", "aime2024", "aime2025"]


def p1(path):
    try:
        rows = list(csv.DictReader(open(path)))
        v = rows[0].get("pass@1") if rows else None
        return ("%.3f" % float(v)) if v not in (None, "") else "-"
    except Exception:
        return "-"


def row(label, base):
    vals = []
    for ds in DS:
        f = os.path.join(base, ds + ".parquet.pass.csv")
        vals.append(p1(f) if os.path.exists(f) else "-")
    print("%-30s %7s %7s %7s %7s" % (label, vals[0], vals[1], vals[2], vals[3]))


def hdr(t):
    print("\n### %s" % t)
    print("%-30s %7s %7s %7s %7s" % ("exp (step105,16K)", "LCBv5", "LCBv6", "AIME24", "AIME25"))


GROUPS = [
    ("1.5B-16K main", [("r1-1.5b-16k", x) for x in ["grpo", "in", "tok", "traj", "prob"]]),
    ("1.5B-16K +KL", [("r1-1.5b-16k-kl", x) for x in ["in-kl", "tok-kl", "traj-kl", "prob-kl"]]),
    ("1.5B-16K +LP", [("r1-1.5b-16k-lp", x) for x in ["in-lp", "tok-lp", "traj-lp", "prob-lp"]]),
    ("1.5B random", [("r1-1.5b-16k-random", "grpo-random")]),
    ("4B-16K main", [("qwen3-4b-16k", x) for x in ["grpo", "in", "tok", "traj", "prob"]]),
    ("7B swe-rl main", [("swe-rl", "swe-rl-" + x) for x in ["grpo", "intuitor", "token_entropy", "trajectory_entropy", "prob_disparity"]]),
    ("7B swe-rl +KL", [("swe-rl-kl", "swe-rl-" + x + "-kl") for x in ["intuitor", "token_entropy", "trajectory_entropy", "prob_disparity"]]),
    ("7B random", [("swe-rl-random", "grpo-random")]),
]

for title, items in GROUPS:
    hdr(title)
    for grp, exp in items:
        row(exp, "%s/%s/%s/global_step_105/actor/hf_model/eval_out_16384" % (O, grp, exp))

hdr("step0 base")
for n in ["r1-1.5b-base", "coder-7b-base"]:
    row(n, "%s/step0/%s/eval_out_16384" % (O, n))
