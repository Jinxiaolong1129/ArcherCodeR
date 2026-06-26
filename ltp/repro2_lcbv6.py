#!/usr/bin/env python3
# Run the REAL run_test on one v6 (code, testcases) pair with debug to get the actual error_code.
import sys, json, re
LCB = "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/Archer_eval/LiveCodeBench"
sys.path.insert(0, LCB)
import pandas as pd
from lcb_runner.evaluation.testing_util import run_test

PARQUET = sys.argv[1]
df = pd.read_parquet(PARQUET)

def extract_answer(s):
    m = re.search(r"```python\n(.*?)```", s, re.DOTALL)
    return m.group(1) if m else s

# problem 0
row = df.iloc[0]
resp = row["responses"][0]
r0 = resp[0] if hasattr(resp, "__getitem__") else resp
code = extract_answer(str(r0))

gt = row["reward_model"]["ground_truth"]
tcs = json.loads(gt) if isinstance(gt, str) else gt
inputs = [tc.get("input", "") for tc in tcs]
outputs = [tc.get("output", "") for tc in tcs]
sample = {"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": None})}

print("num test cases:", len(inputs))
print("first input:", repr(inputs[0]), "expected:", repr(outputs[0]))
print("extracted code (head):", repr(code[:120]))
print("=== running real run_test(debug=True) ... ===")
res, meta = run_test(sample, test=code, debug=True, timeout=6)
print("RESULTS:", res[:10] if isinstance(res, list) else res)
print("META:", {k: (str(v)[:200]) for k, v in (meta or {}).items()})
