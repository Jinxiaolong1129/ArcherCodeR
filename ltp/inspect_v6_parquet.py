#!/usr/bin/env python3
# Inspect a v6 generation parquet: prompt + a generated response, to see why scoring=0.
import sys
P = sys.argv[1]
try:
    import pandas as pd
    df = pd.read_parquet(P)
except Exception as e:
    print("pandas/pyarrow read failed:", e)
    sys.exit(1)
print("rows:", len(df), "cols:", list(df.columns))
r0 = df.iloc[0]
# prompt
pr = r0.get("prompt")
ps = str(pr)
print("\n--- prompt[0] (first 700 chars) ---")
print(ps[:700])
# responses
resp = r0.get("responses")
try:
    first = resp[0]
except Exception:
    first = resp
fs = str(first)
print("\n--- response[0][0] (first 900 chars) ---")
print(fs[:900])
print("\n--- response[0][0] LAST 400 chars (look for ```python / final code) ---")
print(fs[-400:])
print("\n--- response length (chars):", len(fs))
