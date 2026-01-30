#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download MinervaMAth dataset from HuggingFace and convert to Verl format.

Usage:
    conda activate archer
    python tools/download_minervamath_from_hf.py

Source:
    - math-ai/minervamath (272 problems, test split only)

Output:
    - data/test/minervamath.parquet
    - data/test/minervamath.json
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "./data/test"

# System prompt appended to each problem
MATH_SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

# Dataset source on HuggingFace
DATASET_CONFIG = {
    "minervamath": {
        "hf_repo": "math-ai/minervamath",
        "split": "test",
        "problem_field": "question",
        "answer_field": "answer",
    },
}


# ============================================================
# Helper Functions
# ============================================================

def extract_answer(solution: str) -> str:
    """
    Extract answer from \\boxed{...} format.
    If no boxed format found, return the original string.
    
    Examples:
        "\\boxed{204}" -> "204"
        "204" -> "204"
        "4.5e33" -> "4.5e33"
    """
    if solution is None:
        return ""
    solution = str(solution)
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if match:
        return match.group(1).strip()
    return solution.strip()


def create_verl_format(problem: str, answer: str, data_source: str) -> dict:
    """
    Convert a single problem to Verl format.
    
    Args:
        problem: The math problem text
        answer: The correct answer (will be extracted if in boxed format)
        data_source: Dataset identifier (e.g., "minervamath")
    
    Returns:
        Dictionary in Verl format:
        {
            "data_source": "minervamath",
            "prompt": [{"role": "user", "content": "..."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "..."}
        }
    """
    answer_str = extract_answer(answer)
    
    prompt = [
        {"role": "user", "content": f"{problem}\n\n{MATH_SYSTEM_PROMPT}"}
    ]
    
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer_str
        }
    }


def download_and_convert(data_source: str, config: dict, output_dir: str) -> pd.DataFrame:
    """
    Download dataset from HuggingFace and convert to Verl format.
    
    Args:
        data_source: Dataset name (e.g., "minervamath")
        config: Dataset configuration dict
        output_dir: Output directory path
    
    Returns:
        Converted DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing {data_source}")
    print(f"{'='*60}")
    
    # Download from HuggingFace
    print(f"Downloading from {config['hf_repo']}...")
    ds = load_dataset(config['hf_repo'], split=config['split'])
    print(f"  Loaded {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")
    
    # Show sample
    print(f"\n  Sample item:")
    sample = ds[0]
    for k, v in sample.items():
        preview = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
        print(f"    {k}: {preview}")
    
    # Convert to Verl format
    verl_data = []
    for item in ds:
        problem = item[config['problem_field']]
        answer = item[config['answer_field']]
        
        verl_item = create_verl_format(problem, answer, data_source)
        verl_data.append(verl_item)
    
    # Create DataFrame
    df = pd.DataFrame(verl_data)
    
    # Save as parquet
    parquet_path = os.path.join(output_dir, f"{data_source}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved to {parquet_path}")
    
    # Save as JSON (backup)
    json_path = os.path.join(output_dir, f"{data_source}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(verl_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {json_path}")
    
    return df


def verify_dataset(df: pd.DataFrame, data_source: str) -> bool:
    """
    Verify the dataset has correct Verl format.
    
    Args:
        df: DataFrame to verify
        data_source: Expected data_source value
    
    Returns:
        True if valid, False otherwise
    """
    print(f"\n🔍 Verifying {data_source}...")
    
    # Check columns
    required_cols = ["data_source", "prompt", "ability", "reward_model"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  ✗ Missing column: {col}")
            return False
    
    # Check sample
    sample = df.iloc[0]
    
    # Verify data_source
    if sample['data_source'] != data_source:
        print(f"  ✗ Wrong data_source: {sample['data_source']}")
        return False
    
    # Verify prompt format
    prompt = sample['prompt']
    if isinstance(prompt, (list, tuple)):
        prompt = list(prompt)
    if not isinstance(prompt, list) or len(prompt) == 0:
        print(f"  ✗ Invalid prompt format")
        return False
    
    # Verify reward_model
    rm = sample['reward_model']
    if not isinstance(rm, dict) or 'ground_truth' not in rm:
        print(f"  ✗ Invalid reward_model format")
        return False
    
    print(f"  ✓ Format valid!")
    print(f"  ✓ Total samples: {len(df)}")
    print(f"  ✓ Sample ground_truth: {rm['ground_truth']}")
    
    return True


def show_sample_answers(df: pd.DataFrame, data_source: str, n: int = 10):
    """Display first n answers for verification."""
    print(f"\n📊 First {n} answers for {data_source}:")
    for i, row in df.head(n).iterrows():
        gt = row['reward_model']['ground_truth']
        print(f"  Problem {i+1:3d}: {gt}")
    print(f"  ... (total {len(df)} problems)")


def analyze_answer_types(df: pd.DataFrame, data_source: str):
    """Analyze answer types in the dataset."""
    print(f"\n📈 Answer type analysis for {data_source}:")
    
    answers = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]
    
    # Count different types
    int_count = sum(1 for a in answers if a.isdigit() or (a.startswith('-') and a[1:].isdigit()))
    float_count = sum(1 for a in answers if '.' in a and not 'e' in a.lower())
    sci_count = sum(1 for a in answers if 'e' in a.lower())
    frac_count = sum(1 for a in answers if '/' in a)
    other_count = len(answers) - int_count - float_count - sci_count - frac_count
    
    print(f"  Integer answers: {int_count}")
    print(f"  Float answers: {float_count}")
    print(f"  Scientific notation: {sci_count}")
    print(f"  Fractions: {frac_count}")
    print(f"  Other: {other_count}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("MinervaMAth Dataset Downloader for Verl")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each dataset
    results = {}
    for data_source, config in DATASET_CONFIG.items():
        try:
            df = download_and_convert(data_source, config, OUTPUT_DIR)
            verify_dataset(df, data_source)
            show_sample_answers(df, data_source)
            analyze_answer_types(df, data_source)
            results[data_source] = len(df)
        except Exception as e:
            print(f"  ✗ Error processing {data_source}: {e}")
            import traceback
            traceback.print_exc()
            results[data_source] = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for ds, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {ds}: {count} samples")
    
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

