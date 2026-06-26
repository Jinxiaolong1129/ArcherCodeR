#!/usr/bin/env python3
"""Download all archer-internal-rl wandb runs' history+summary to flat CSVs (run on cpu-2).

Output: ltp/archer_wandb_data/
  <key>.history.csv   full per-step metric time series
  <key>.summary.json  final summary dict
  manifest.json       project, metric list, run roster with bucket/model/method/ablation

<key> == wandb run name == "<model>-<method>[-<ablation>]" (already the unified key).
Then rsync archer_wandb_data/ back to the Mac analysis tree and plot there.
"""
import json
import os

import wandb

PROJECT = "swe-prm/archer-internal-rl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archer_wandb_data")

# Metrics we care about (superset; missing ones come back as NaN per run).
METRICS = [
    "val-core/livecodebench/acc/mean@4",
    "val-aux/livecodebench/acc/best@4/mean",
    "val-aux/livecodebench/acc/worst@4/mean",
    "response_length/mean",
    "val_response_length/mean",
    "actor/action_reward",
    "actor/ppo_kl",
    "actor/entropy",
    "external_reward/mean",
    "internal_metrics/trajectory_entropy/mean",
    "internal_metrics/token_entropy/mean",
    "internal_metrics/self_certainty/mean",
    "internal_metrics/prob_disparity/mean",
    "val_token/thinking_and_reasoning",
    "val_token/repetition_ratio",
]

# Methods present in the run-name token after the model.
METHODS = {"grpo", "in", "tok", "traj", "prob"}


def parse_key(name):
    """coder7b-tok-kl -> (model='coder7b', method='tok', ablation='kl')."""
    # model prefix is everything up to the first method token
    parts = name.split("-")
    model_toks, rest = [], list(parts)
    while rest and rest[0] not in METHODS and rest[0] != "grpo":
        model_toks.append(rest.pop(0))
    model = "-".join(model_toks)
    method = rest.pop(0) if rest else "?"
    ablation = "-".join(rest) if rest else ""        # "", "kl", "lenpen", "rand"
    if ablation == "rand":
        ablation = "random"
    # bucket assignment
    if ablation == "kl":
        bucket = "ablation-kl"
    elif ablation == "lenpen":
        bucket = "ablation-lenpen"
    elif ablation == "random":
        bucket = "baseline-random"
    elif model in ("r1-1.5b8k", "r1-1.5b12k"):
        bucket = "length-sweep"
    else:
        bucket = "main"
    return model, method, ablation, bucket


def main():
    os.makedirs(OUT, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs = list(api.runs(PROJECT))
    roster = []
    for r in sorted(runs, key=lambda x: x.name):
        key = r.name
        model, method, ablation, bucket = parse_key(key)
        df = r.history(samples=5000)        # full history (105 steps << 5000)
        if "_step" in df.columns:
            df = df.sort_values("_step")
        hist_path = os.path.join(OUT, f"{key}.history.csv")
        df.to_csv(hist_path, index=False)
        summ_path = os.path.join(OUT, f"{key}.summary.json")
        with open(summ_path, "w") as f:
            json.dump(dict(r.summary), f, indent=2, default=str)
        roster.append({
            "key": key, "model": model, "method": method,
            "ablation": ablation, "bucket": bucket,
            "id": r.id, "state": r.state,
            "history_rows": int(len(df)),
            "url": r.url,
        })
        print(f"OK  {bucket:16s} {key:24s} rows={len(df):4d} cols={df.shape[1]}")

    manifest = {
        "project_path": PROJECT,
        "metrics": METRICS,
        "n_runs": len(roster),
        "runs": roster,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(roster)} runs -> {OUT}")


if __name__ == "__main__":
    main()
