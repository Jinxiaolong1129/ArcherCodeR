#!/usr/bin/env python3
"""Local math scorer for AIME-style eval. Reads the main_generation output parquet
(prompt + responses[n_samples]) + the dataset json (reward_model.ground_truth = answer),
extracts each response's \\boxed{} answer and grades vs ground_truth. Reports avg@k & pass@k.

Reuses rewards/math_utils (same grader as the training reward). CPU-only, no GPU needed.

Usage: python tools/compute_math_metrics_local.py --eval_file <gen.parquet> --testcase_file <dataset.json>
"""
import sys, os, json, argparse
import polars as pl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from rewards.math_utils.utils import grade_answer_sympy, grade_answer_mathd, extract_answer as math_extract_answer

def grade(resp, gt):
    ans = math_extract_answer(resp)
    if ans is None or not isinstance(ans, str):
        return False
    gt = str(gt)
    try:
        if grade_answer_mathd(ans, gt):
            return True
    except Exception:
        pass
    try:
        return bool(grade_answer_sympy(ans, gt))
    except Exception:
        return False

ap = argparse.ArgumentParser()
ap.add_argument("--eval_file", required=True)
ap.add_argument("--testcase_file", required=True)
ap.add_argument("--num_workers", type=int, default=1)  # math grading is light; serial is fine
args = ap.parse_args()

df = pl.read_parquet(args.eval_file)
prompt2responses = {}
for i in range(len(df)):
    p = df[i]["prompt"][0][0]["content"]
    prompt2responses[p] = list(df[i]["responses"][0])

data = json.load(open(args.testcase_file))
n_prob = 0
sum_avg = 0.0   # avg@k (mean per-sample correctness)
sum_passk = 0   # pass@k (any correct)
matched = 0
for item in data:
    qc = item["prompt"][0]["content"]
    gt = item["reward_model"]["ground_truth"]
    resp = prompt2responses.get(qc)
    if resp is None:
        for p, r in prompt2responses.items():
            if qc[:500] in p or p[:500] in qc:
                resp = r; break
    if resp is None:
        continue
    matched += 1
    corrects = [grade(r, gt) for r in resp]
    k = len(corrects)
    sum_avg += sum(corrects) / k
    sum_passk += 1 if any(corrects) else 0
    n_prob += 1

print(f"matched {matched}/{len(data)} problems | k(samples/problem)={k}")
print(f"avg@{k} (mean accuracy) = {100*sum_avg/n_prob:.2f}%")
print(f"pass@{k}                = {100*sum_passk/n_prob:.2f}%")
