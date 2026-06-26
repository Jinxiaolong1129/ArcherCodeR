#!/usr/bin/env python3
# Final corrected table: LCB v5/v6 from .pass.lcb.local.csv (authoritative), AIME from .pass.csv.
import csv, os
O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"


def col(path, colname):
    if not os.path.exists(path):
        return "-"
    try:
        r = list(csv.DictReader(open(path)))
        v = r[0].get(colname) if r else None
        return ("%.3f" % float(v)) if v not in (None, "") else "-"
    except Exception:
        return "-"


def lcb(base, ds):
    return col(os.path.join(base, ds + ".parquet.pass.lcb.local.csv"), "pass@1")


def aime(base, ds):
    return col(os.path.join(base, ds + ".parquet.pass.csv"), "pass@1")


def row(label, base):
    print("%-28s %7s %7s %7s %7s" % (label,
          lcb(base, "livecodebench_v5"), lcb(base, "livecodebench_v6"),
          aime(base, "aime2024"), aime(base, "aime2025")))


def hdr(t):
    print("\n### %s" % t)
    print("%-28s %7s %7s %7s %7s" % ("exp", "LCBv5", "LCBv6", "AIME24", "AIME25"))


G = [
    ("1.5B-16K main", "r1-1.5b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("1.5B-16K +KL", "r1-1.5b-16k-kl", ["in-kl", "tok-kl", "traj-kl", "prob-kl"]),
    ("1.5B-16K +LP", "r1-1.5b-16k-lp", ["in-lp", "tok-lp", "traj-lp", "prob-lp"]),
    ("1.5B random", "r1-1.5b-16k-random", ["grpo-random"]),
    ("4B-16K main", "qwen3-4b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("7B swe-rl main", "swe-rl", ["swe-rl-grpo", "swe-rl-intuitor", "swe-rl-token_entropy", "swe-rl-trajectory_entropy", "swe-rl-prob_disparity"]),
    ("7B swe-rl +KL", "swe-rl-kl", ["swe-rl-intuitor-kl", "swe-rl-token_entropy-kl", "swe-rl-trajectory_entropy-kl", "swe-rl-prob_disparity-kl"]),
    ("7B random", "swe-rl-random", ["grpo-random"]),
]
for title, grp, exps in G:
    hdr(title)
    for e in exps:
        row(e, "%s/%s/%s/global_step_105/actor/hf_model/eval_out_16384" % (O, grp, e))
hdr("step0 base")
for n in ["r1-1.5b-base", "coder-7b-base"]:
    row(n, "%s/step0/%s/eval_out_16384" % (O, n))
