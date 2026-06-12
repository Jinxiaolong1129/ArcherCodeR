#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download LiveCodeBench v6 and convert to Verl eval format.

Output files:
  - Archer_eval/data/test/livecodebench_v6.json
  - Archer_eval/data/test/livecodebench_v6.parquet

Usage:
  cd Archer_eval
  python tools/download_livecodebench_v6.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def _load_v6_dataset():
    """
    Load LCB v6 from Hugging Face.
    Prefer datasets API; if version selection is not honored, fallback to main/test6.jsonl.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    repo_id = "livecodebench/code_generation_lite"

    # Try multiple config styles to improve compatibility across datasets versions.
    configs_to_try = [
        {"name": "release_latest", "version_tag": "release_v6"},
        {"version_tag": "release_v6"},
        {"name": "release_v6"},
        {},
    ]

    for config in configs_to_try:
        try:
            print(f"[INFO] Trying load_dataset config: {config}")
            ds = load_dataset(repo_id, split="test", trust_remote_code=True, **config)
            rows = [dict(x) for x in ds]
            if rows:
                print(f"[OK] Loaded {len(rows)} rows via datasets API")
                return rows
        except Exception as e:
            print(f"[WARN] load_dataset failed with {config}: {e}")

    # Fallback: direct download of test6.jsonl from main branch.
    print("[INFO] Falling back to hf_hub_download: test6.jsonl@main")
    jsonl_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="test6.jsonl",
        revision="main",
    )

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[OK] Loaded {len(rows)} rows via hf_hub_download fallback")
    return rows


def _parse_date(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def _to_verl_item(raw_item: dict) -> dict:
    """
    Convert one LCB raw item to Verl format expected by Archer_eval generation scripts.
    """
    question = raw_item.get("question_content", "")
    if not isinstance(question, str):
        question = str(question)

    public_tests = raw_item.get("public_test_cases", "[]")
    private_tests = raw_item.get("private_test_cases", "[]")

    # Keep the original serialized format in ground_truth for maximum compatibility.
    ground_truth = {
        "public_test_cases": public_tests,
        "private_test_cases": private_tests,
        "question_id": raw_item.get("question_id", ""),
        "question_title": raw_item.get("question_title", ""),
        "platform": raw_item.get("platform", ""),
        "contest_id": raw_item.get("contest_id", ""),
        "contest_date": raw_item.get("contest_date", ""),
        "starter_code": raw_item.get("starter_code", ""),
        "difficulty": raw_item.get("difficulty", ""),
        "metadata": raw_item.get("metadata", "{}"),
    }

    return {
        "data_source": "livecodebench_v6",
        "prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        },
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]  # Archer_eval/
    output_dir = base_dir / "data" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / "livecodebench_v6.json"
    out_parquet = output_dir / "livecodebench_v6.parquet"

    print("=" * 60)
    print("Download and Convert LiveCodeBench v6 for Verl")
    print("=" * 60)

    rows = _load_v6_dataset()

    # Keep official v6 evaluation window: [2025-02-01, 2025-05-01].
    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 5, 1)

    filtered = []
    for item in rows:
        contest_date = _parse_date(item.get("contest_date", ""))
        if contest_date is None:
            continue
        if start_date <= contest_date <= end_date:
            filtered.append(item)

    print(f"[INFO] Raw rows: {len(rows)}")
    print(f"[INFO] Filtered v6 rows (2025-02-01 ~ 2025-05-01): {len(filtered)}")

    verl_rows = [_to_verl_item(x) for x in filtered]
    df = pd.DataFrame(verl_rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verl_rows, f, ensure_ascii=False, indent=2)

    df.to_parquet(out_parquet, index=False)

    print(f"[OK] Saved JSON   : {out_json}")
    print(f"[OK] Saved Parquet: {out_parquet}")
    print(f"[OK] Total samples: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
