#!/usr/bin/env python3
# Correct LCB results: read .pass.lcb.local.csv (authoritative scorer output), not .pass.csv.
import csv, os, glob
O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"


def lcb_p1(base, ds):
    f = os.path.join(base, ds + ".parquet.pass.lcb.local.csv")
    if not os.path.exists(f):
        return "-"
    try:
        r = list(csv.DictReader(open(f)))
        return ("%.3f" % float(r[0]["pass@1"])) if r else "-"
    except Exception:
        return "-"


EXPS = [
    ("r1-1.5b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("r1-1.5b-16k-kl", ["in-kl", "tok-kl", "traj-kl", "prob-kl"]),
    ("r1-1.5b-16k-lp", ["in-lp", "tok-lp", "traj-lp", "prob-lp"]),
    ("r1-1.5b-16k-random", ["grpo-random"]),
    ("qwen3-4b-16k", ["grpo", "in", "tok", "traj", "prob"]),
    ("swe-rl", ["swe-rl-grpo", "swe-rl-intuitor", "swe-rl-token_entropy", "swe-rl-trajectory_entropy", "swe-rl-prob_disparity"]),
    ("swe-rl-kl", ["swe-rl-intuitor-kl", "swe-rl-token_entropy-kl", "swe-rl-trajectory_entropy-kl", "swe-rl-prob_disparity-kl"]),
    ("swe-rl-random", ["grpo-random"]),
]
print("%-40s %8s %8s" % ("exp (step105,16K)", "LCBv5", "LCBv6"))
for grp, exps in EXPS:
    for e in exps:
        base = "%s/%s/%s/global_step_105/actor/hf_model/eval_out_16384" % (O, grp, e)
        print("%-40s %8s %8s" % (f"{grp}/{e}", lcb_p1(base, "livecodebench_v5"), lcb_p1(base, "livecodebench_v6")))
for n in ["r1-1.5b-base", "coder-7b-base"]:
    base = "%s/step0/%s/eval_out_16384" % (O, n)
    print("%-40s %8s %8s" % (f"step0/{n}", lcb_p1(base, "livecodebench_v5"), lcb_p1(base, "livecodebench_v6")))
