#!/usr/bin/env python3
# Recompute LCB pass@1 from results.json (no re-run) and compare to the (stale) pass.csv.
import json, os, csv, glob

O = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/output"


def pass1_from_results(results_path):
    """results.json = [[per-sample value] per problem]; sample passes if value==1/True."""
    try:
        data = json.load(open(results_path))
    except Exception as e:
        return None
    if not data:
        return None
    prob_rates = []
    for samples in data:
        if not isinstance(samples, list) or not samples:
            continue
        passed = sum(1 for s in samples if s == 1 or s is True or s == 1.0)
        prob_rates.append(passed / len(samples))
    return sum(prob_rates) / len(prob_rates) if prob_rates else None


def csv_pass1(csv_path):
    try:
        r = list(csv.DictReader(open(csv_path)))
        return r[0].get("pass@1") if r else None
    except Exception:
        return None


# scan all v6 results.json under output, at step105 / eval_out_16384
print("%-44s %10s %10s" % ("exp/step", "csv_pass@1", "results.json_pass@1"))
for rp in sorted(glob.glob(O + "/*/*/global_step_105/actor/hf_model/eval_out_16384/livecodebench_v6.parquet.results.json")):
    cp = rp.replace(".results.json", ".pass.csv")
    label = rp.replace(O + "/", "").replace("/actor/hf_model/eval_out_16384/livecodebench_v6.parquet.results.json", "")
    rj = pass1_from_results(rp)
    cv = csv_pass1(cp)
    print("%-44s %10s %10s" % (label, cv, ("%.3f" % rj) if rj is not None else "None"))
