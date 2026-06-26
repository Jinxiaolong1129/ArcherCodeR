#!/usr/bin/env python3
# Batch-sync all 31 training offline runs to wandb project archer-internal-rl with model-inclusive names.
import glob, os, subprocess, sys

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"
KEYFILE = "/mnt/cephfs/data/processing/xiaolong.jin/code/wandb"
PROJECT = "archer-internal-rl"

os.environ["WANDB_API_KEY"] = open(KEYFILE).read().strip()

PREFIX = {
    "swe-rl": "coder7b", "swe-rl-kl": "coder7b", "swe-rl-random": "coder7b",
    "r1-1.5b-16k": "r1-1.5b16k", "r1-1.5b-16k-kl": "r1-1.5b16k", "r1-1.5b-16k-lp": "r1-1.5b16k",
    "r1-1.5b-16k-random": "r1-1.5b16k", "r1-1.5b-8k-random": "r1-1.5b8k", "r1-1.5b-12k": "r1-1.5b12k",
    "qwen3-4b-16k": "qwen3-4b",
}
METH = {"trajectory_entropy": "traj", "token_entropy": "tok", "prob_disparity": "prob",
        "intuitor": "in", "grpo-random": "grpo-rand", "grpo-12k": "grpo"}


def newname(group, exp):
    pref = PREFIX.get(group, group)
    e = exp
    if e.startswith("swe-rl-"):
        e = e[len("swe-rl-"):]
    for long, short in METH.items():
        if e == long:
            e = short
    for long, short in METH.items():
        e = e.replace(long, short)
    e = e.replace("-lp", "-lenpen")
    return f"{pref}-{e}"


rows = []
for d in sorted(glob.glob(O + "/*/*/wandb/offline-run-*")):
    if not os.path.isdir(d):
        continue
    parts = d.replace(O + "/", "").split("/")
    rows.append((newname(parts[0], parts[1]), d))

print(f"Syncing {len(rows)} runs to project '{PROJECT}' ...\n", flush=True)
ok, fail = 0, 0
for i, (name, d) in enumerate(rows, 1):
    print(f"[{i}/{len(rows)}] {name} ...", flush=True)
    r = subprocess.run(
        ["python3", "-m", "wandb", "sync", "--project", PROJECT, "--id", name, d],
        capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    if "done." in out.lower() or "Syncing:" in out:
        ok += 1
        print(f"    OK {name}", flush=True)
    else:
        fail += 1
        print(f"    FAIL {name}: {out.strip()[-200:]}", flush=True)
print(f"\nDONE: {ok} ok, {fail} fail")
