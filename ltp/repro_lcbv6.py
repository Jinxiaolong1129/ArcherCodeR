#!/usr/bin/env python3
# Repro the grade_stdio code transform on a real stdin solution to find why v6 scores 0.
import sys, ast
sys.path.insert(0, "/mnt/cephfs/data/processing/xiaolong.jin/code/ArcherCodeR/Archer_eval/LiveCodeBench")
from lcb_runner.evaluation.testing_util import clean_if_name, make_function

code = '''import sys

def main():
    X = int(input().strip())
    total_sum = sum(i * j for i in range(1, 10) for j in range(1, 10))
    count_of_X = 0
    for i in range(1, 10):
        for j in range(1, 10):
            if i * j == X:
                count_of_X += 1
    sum_not_X = total_sum - (X * count_of_X)
    print(sum_not_X)

if __name__ == "__main__":
    main()
'''

print("===== after clean_if_name =====")
c1 = clean_if_name(code)
print(c1)
print("===== after make_function =====")
c2 = make_function(c1)
print(c2)

print("===== does it call wrapped_function and produce output for input '1' (expect 2024)? =====")
import io
from unittest.mock import patch, mock_open
g = {}
try:
    exec(compile(c2, "<sol>", "exec"), g)
    wf = g.get("wrapped_function")
    print("wrapped_function defined:", wf is not None)
    inp = "1"
    out = io.StringIO()
    with patch("sys.stdin", io.StringIO(inp)), patch("builtins.input", lambda *a: inp):
        old = sys.stdout
        sys.stdout = out
        try:
            wf()
        finally:
            sys.stdout = old
    print("CAPTURED OUTPUT:", repr(out.getvalue()))
    print("EXPECTED:", repr("2024"))
except Exception as e:
    import traceback
    print("EXEC ERROR:", repr(e))
    traceback.print_exc()
