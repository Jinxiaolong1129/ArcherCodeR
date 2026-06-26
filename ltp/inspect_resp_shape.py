#!/usr/bin/env python3
# Compare the 'responses' nesting/shape between v5 and v6 parquets (scorer does responses[0]).
import sys
import pandas as pd
for P in sys.argv[1:]:
    print("\n=====", P.split("/")[-1], "=====")
    df = pd.read_parquet(P)
    r = df.iloc[0]["responses"]
    print("type(responses):", type(r).__name__, "len:", len(r))
    r0 = r[0]
    print("type(responses[0]):", type(r0).__name__,
          ("len:" + str(len(r0))) if hasattr(r0, "__len__") else "")
    # is responses[0] a full response string, or a list of responses?
    if isinstance(r0, str):
        print("  -> responses[0] is a STRING. head:", repr(r0[:60]))
        print("  -> scorer's `for x in responses[0]` would iterate CHARS (BUG if so)")
    else:
        r00 = r0[0]
        print("  -> responses[0] is a LIST/array. responses[0][0] type:", type(r00).__name__,
              "head:", repr(str(r00)[:60]))
