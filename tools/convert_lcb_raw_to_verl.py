#!/usr/bin/env python3
"""Convert RAW LiveCodeBench (livecodebench/code_generation_lite) json -> verl prompt format
matching the eval data schema. Self-contained: inlines lcb_runner's test-case decode + the
official code-generation prompt (no lcb_runner import, so no anthropic/few-shot-file deps).

Usage: python tools/convert_lcb_raw_to_verl.py <raw_in.json> <verl_out.json>
"""
import json, sys, zlib, pickle, base64

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem "
          "specification) and will generate a correct Python program that matches the "
          "specification and passes all tests.")
FMT_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FMT_WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout "
                       "(do not directly test on the sample inputs). Enclose your code within "
                       "delimiters as follows. Ensure that when the python program runs, it reads "
                       "the inputs, runs the algorithm and writes output to STDOUT.")

def build_prompt(question_content, starter_code):
    p = f"### Question:\n{question_content}\n\n"
    if starter_code:
        p += f"### Format: {FMT_WITH_STARTER}\n```python\n{starter_code}\n```\n\n"
    else:
        p += f"### Format: {FMT_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return SYSTEM + "\n\n" + p

def decode_tests(raw):
    pub = json.loads(raw["public_test_cases"])  # list of {input,output,testtype}
    pv = raw["private_test_cases"]
    try:
        priv = json.loads(pv)
    except Exception:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(pv.encode("utf-8")))))
    return pub + priv

raw_path, out_path = sys.argv[1], sys.argv[2]
# accept either a JSON array or JSONL (one object per line, e.g. test6.jsonl)
_txt = open(raw_path).read().lstrip()
if _txt[:1] == "[":
    raw = json.loads(_txt)
else:
    raw = [json.loads(l) for l in _txt.splitlines() if l.strip()]
out, skipped = [], 0
for i, r in enumerate(raw):
    try:
        content = build_prompt(r["question_content"], r.get("starter_code", ""))
        tests = decode_tests(r)
        # normalize test dicts to {input, output, testtype}
        gt = [{"input": t["input"], "output": t["output"], "testtype": t.get("testtype", "stdin")} for t in tests]
        out.append({
            "data_source": "livecodebench",
            "prompt": [{"role": "user", "content": content}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"split": "test", "index": i, "reference": None, "question_id": r.get("question_id")},
        })
    except Exception as e:
        skipped += 1
        if skipped <= 3:
            print(f"[skip {i}] {type(e).__name__}: {e}")
json.dump(out, open(out_path, "w"))
print(f"wrote {len(out)} (skipped {skipped}) -> {out_path}")
