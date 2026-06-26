#!/usr/bin/env python3
# Build the proposed wandb run-name mapping for consolidating all training runs into one project.
import glob, os
O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"

# group -> (model/length prefix, suffix)
PREFIX = {
    "swe-rl": "coder7b",
    "swe-rl-kl": "coder7b",
    "swe-rl-random": "coder7b",
    "r1-1.5b-16k": "r1-1.5b16k",
    "r1-1.5b-16k-kl": "r1-1.5b16k",
    "r1-1.5b-16k-lp": "r1-1.5b16k",
    "r1-1.5b-16k-random": "r1-1.5b16k",
    "r1-1.5b-8k-random": "r1-1.5b8k",
    "r1-1.5b-12k": "r1-1.5b12k",
    "qwen3-4b-16k": "qwen3-4b",
}
# normalize method tokens
METH = {"trajectory_entropy": "traj", "token_entropy": "tok", "prob_disparity": "prob",
        "intuitor": "in", "grpo-random": "grpo-rand", "grpo-12k": "grpo"}


def newname(group, exp):
    pref = PREFIX.get(group, group)
    e = exp
    # strip leading "swe-rl-" from 7b exp names
    if e.startswith("swe-rl-"):
        e = e[len("swe-rl-"):]
    # apply method normalization on the core token (keep -kl/-lp suffix)
    for long, short in METH.items():
        if e == long:
            e = short
    # handle method-with-suffix like trajectory_entropy-kl
    for long, short in METH.items():
        e = e.replace(long, short)
    # clearer suffix for length penalty
    e = e.replace("-lp", "-lenpen")
    return f"{pref}-{e}"


rows = []
for d in sorted(glob.glob(O + "/*/*/wandb/offline-run-*")):
    if not os.path.isdir(d):
        continue
    parts = d.replace(O + "/", "").split("/")
    group, exp = parts[0], parts[1]
    rows.append((group, exp, newname(group, exp), d))

print("%-22s %-22s -> %-18s" % ("group", "exp", "new wandb name"))
for g, e, nn, d in rows:
    print("%-22s %-22s -> %-18s" % (g, e, nn))
print("\ntotal runs:", len(rows))
