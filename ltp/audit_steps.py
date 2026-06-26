#!/usr/bin/env python3
# Audit which (step, dataset) cells are SCORED for every experiment.
# Generation = .parquet exists; Scored = .pass.lcb.local.csv (LCB) or .pass.csv (AIME).
import os, glob

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"
STEPS = ["20", "50", "80", "105"]
DS = [("livecodebench_v5", "lcb"), ("livecodebench_v6", "lcb"),
      ("aime2024", "aime"), ("aime2025", "aime")]
LEN = "16384"

GROUPS = [
    ("1.5B main", "r1-1.5b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("1.5B +KL", "r1-1.5b-16k-kl", ["in-kl", "tok-kl", "traj-kl", "prob-kl"]),
    ("1.5B +LP", "r1-1.5b-16k-lp", ["in-lp", "tok-lp", "traj-lp", "prob-lp"]),
    ("1.5B random", "r1-1.5b-16k-random", ["grpo-random"]),
    ("4B main", "qwen3-4b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("7B main", "swe-rl", ["swe-rl-grpo", "swe-rl-intuitor", "swe-rl-token_entropy",
                           "swe-rl-trajectory_entropy", "swe-rl-prob_disparity"]),
    ("7B +KL", "swe-rl-kl", ["swe-rl-intuitor-kl", "swe-rl-token_entropy-kl",
                             "swe-rl-trajectory_entropy-kl", "swe-rl-prob_disparity-kl"]),
    ("7B random", "swe-rl-random", ["grpo-random"]),
]


def base(grp, exp, step):
    return f"{O}/{grp}/{exp}/global_step_{step}/actor/hf_model/eval_out_{LEN}"


def gen_exists(grp, exp, step, ds):
    return os.path.exists(f"{base(grp,exp,step)}/{ds}.parquet")


def scored(grp, exp, step, ds, kind):
    suf = ".pass.lcb.local.csv" if kind == "lcb" else ".pass.csv"
    return os.path.exists(f"{base(grp,exp,step)}/{ds}.parquet{suf}")


print("Legend: ✓=scored  g=generated-only(not scored)  .=missing   | steps order: 20 50 80 105\n")
hdr = "%-32s %-12s %-12s %-12s %-12s" % ("exp", "LCBv5", "LCBv6", "AIME24", "AIME25")
for title, grp, exps in GROUPS:
    print(f"### {title}")
    print(hdr)
    for exp in exps:
        cells = []
        for ds, kind in DS:
            s = ""
            for st in STEPS:
                if scored(grp, exp, st, ds, kind):
                    s += "✓"
                elif gen_exists(grp, exp, st, ds):
                    s += "g"
                else:
                    s += "."
            cells.append(s)
        print("%-32s %-12s %-12s %-12s %-12s" % (exp, *cells))
    print()
