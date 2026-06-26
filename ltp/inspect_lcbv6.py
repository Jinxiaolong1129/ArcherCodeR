#!/usr/bin/env python3
# Diagnose LCB-v6 all-zero: inspect ground_truth (test cases) in v6.json vs v6_raw.json.
import json, sys

T = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/data/test"


def load(name):
    p = f"{T}/{name}"
    print(f"\n===== {name} =====")
    try:
        d = json.load(open(p))
    except Exception as e:
        print("  load error:", e)
        return
    print("  type:", type(d).__name__)
    if isinstance(d, dict):
        keys = list(d.keys())
        print("  columns:", keys)
        # columnar dict-of-dicts: d[col] = {idx: val}
        rm = d.get("reward_model")
        if rm is not None:
            idxs = list(rm.keys()) if isinstance(rm, dict) else range(len(rm))
            n = len(idxs)
            print("  num problems:", n)
            # inspect first 2 ground_truths
            for i in list(idxs)[:2]:
                gt = rm[i] if isinstance(rm, dict) else rm[i]
                gt = gt.get("ground_truth") if isinstance(gt, dict) else gt
                s = json.dumps(gt) if not isinstance(gt, str) else gt
                print(f"  [{i}] ground_truth len={len(s)} head={s[:200]}")
    elif isinstance(d, list):
        print("  num problems:", len(d))
        for x in d[:2]:
            gt = x.get("reward_model", {}).get("ground_truth") if isinstance(x, dict) else None
            s = json.dumps(gt) if not isinstance(gt, str) else (gt or "")
            print(f"  ground_truth len={len(s)} head={s[:200]}")


for name in sys.argv[1:]:
    load(name)
