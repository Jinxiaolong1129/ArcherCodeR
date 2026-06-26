#!/usr/bin/env python3
# Fix wandb DISPLAY names: --id set the run id, but the display name stayed the training-time name.
# Set each run's display .name = its id (the model-inclusive name).
import glob, os, wandb

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"
KEYFILE = "/mnt/cephfs/data/processing/xiaolong.jin/code/wandb"
ENTITY, PROJECT = "swe-prm", "archer-internal-rl"
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
    e = exp[len("swe-rl-"):] if exp.startswith("swe-rl-") else exp
    for lo, sh in METH.items():
        if e == lo:
            e = sh
    for lo, sh in METH.items():
        e = e.replace(lo, sh)
    return f"{pref}-{e.replace('-lp', '-lenpen')}"


ids = []
for d in sorted(glob.glob(O + "/*/*/wandb/offline-run-*")):
    if os.path.isdir(d):
        p = d.replace(O + "/", "").split("/")
        ids.append(newname(p[0], p[1]))

api = wandb.Api()
ok = fail = 0
for rid in ids:
    try:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        old = r.name
        r.name = rid
        r.update()
        ok += 1
        print(f"  {old}  ->  {rid}", flush=True)
    except Exception as e:
        fail += 1
        print(f"  FAIL {rid}: {str(e)[:120]}", flush=True)
print(f"\nrenamed: {ok} ok, {fail} fail")
