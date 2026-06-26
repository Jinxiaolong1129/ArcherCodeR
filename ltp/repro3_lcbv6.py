#!/usr/bin/env python3
# Faithful repro of the REAL scorer path: polars read + its extract_answer + real run_test.
import sys, json
LCB = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/Archer_eval/LiveCodeBench"
sys.path.insert(0, LCB)
import polars as pl
from lcb_runner.evaluation.testing_util import run_test
sys.path.insert(0, LCB + "/lcb_runner/evaluation")
import compute_code_generation_metrics_local as M

PARQUET = sys.argv[1]
TC = sys.argv[2]   # testcase_file (v6.json)

df = pl.read_parquet(PARQUET)
resp_list = df[0]['responses'][0]   # polars: list of n_samples for row 0
print("n responses row0:", len(resp_list))
r0 = resp_list[0]
print("THOUGHT_DELIMITER_END =", repr(getattr(M, "THOUGHT_DELIMITER_END", "<none>")))
de = getattr(M, "THOUGHT_DELIMITER_END", "</think>")
model_solution = r0 if de not in r0 else r0.split(de)[1]
code = M.extract_answer(model_solution)
print("extracted solution_code head:", repr((code or "")[:120]))
print("extracted solution_code len:", len(code or ""))

# build sample from v6 ground_truth (problem 0)
data = json.load(open(TC))
gt = data[0]["reward_model"]["ground_truth"]
tcs = json.loads(gt) if isinstance(gt, str) else gt
inputs = [t.get("input","") for t in tcs]; outputs=[t.get("output","") for t in tcs]
sample = {"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": None})}
print("=== real run_test on correctly-extracted code ===")
res, meta = run_test(sample, test=code, debug=False, timeout=6)
print("RESULTS[:5]:", res[:5] if isinstance(res,list) else res)
print("META:", {k:str(v)[:160] for k,v in (meta or {}).items()})
