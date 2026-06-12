#!/usr/bin/env python

# Archer_eval/LiveCodeBench/lcb_runner/evaluation/compute_certainty_correctness.py


# -*- coding: utf-8 -*-
"""
Compute LCB correctness for certainty parquet files.
- Evaluate each response's correctness
- Update is_correct column in certainty parquet
- Compute pass@1, pass@4, pass@8 metrics

Usage:
    python compute_certainty_correctness.py --certainty_file <path> --lcb_version v5
    python compute_certainty_correctness.py --certainty_file <path> --lcb_version v6 --save
    python compute_certainty_correctness.py --batch --base_dir <dir> --lcb_version v5
"""

import os
import sys

# Setup paths before imports (like run_lcb_eval_v5.sh)
# Script: .../LiveCodeBench/lcb_runner/evaluation/compute_certainty_correctness.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../lcb_runner/evaluation
_LCB_RUNNER_DIR = os.path.dirname(_SCRIPT_DIR)            # .../lcb_runner
_LCB_DIR = os.path.dirname(_LCB_RUNNER_DIR)               # .../LiveCodeBench
_BASE_DIR = os.path.dirname(_LCB_DIR)                     # .../Archer_eval

# Add LiveCodeBench to path for lcb_runner imports
if _LCB_DIR not in sys.path:
    sys.path.insert(0, _LCB_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import csv
import argparse
import multiprocessing
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
from lcb_runner.evaluation.testing_util import run_test

THOUGHT_DELIMITER_END = "</think>"


# ============ Functions copied from compute_code_generation_metrics_online.py ============
# (Avoid importing that file because it has parser.parse_args() at module level)

def extract_answer(model_solution):
    """Extract Python code from response"""
    pattern = r"```python\n(.*?)```"
    match = re.findall(pattern, model_solution, re.DOTALL)
    if len(match) == 0:
        return None
    else:
        return match[-1]


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=min((timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5, 900)
    )

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list[0] if metadata_list else {}


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(
                sample, o, timeout=timeout, debug=debug
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            res.append(curr_res)
            metadata.append(curr_metadata if 'curr_metadata' in dir() else {})
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """Evaluate code generations against test cases."""
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs), desc="Evaluating") as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 4, 8],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    """Compute pass@k metrics for code generations."""
    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        for generation in generation_list:
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)} samples...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    return [metrics, results, metadatas]
# ============ End of copied functions ============

# LCB Version configs
LCB_VERSION_CONFIG = {
    "v5": {"release_version": "release_v5", "start_date": "2024-08-01", "end_date": "2025-02-01"},
    "v6": {"release_version": "release_v6", "start_date": "2025-02-01", "end_date": None},
}


def process_response(response):
    """Extract code from response"""
    if THOUGHT_DELIMITER_END not in response:
        model_solution = response
    else:
        model_solution = response.split(THOUGHT_DELIMITER_END)[1]
    
    code = extract_answer(model_solution)
    if code is None or not isinstance(code, str):
        code = ''
    return code


def load_certainty_and_original(certainty_path):
    """
    Load certainty parquet and corresponding original parquet.
    
    certainty_path: .../livecodebench_v5_certainty.parquet
    original_path:  .../livecodebench_v5.parquet
    """
    original_path = str(certainty_path).replace('_certainty.parquet', '.parquet')
    
    print(f"Loading certainty: {certainty_path}")
    certainty_df = pd.read_parquet(certainty_path)
    print(f"  Certainty samples: {len(certainty_df)}")
    
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original parquet not found: {original_path}")
    
    print(f"Loading original: {original_path}")
    original_df = pl.read_parquet(original_path)
    print(f"  Original rows: {len(original_df)}")
    
    return certainty_df, original_df


def match_to_lcb_dataset(original_df, lcb_dataset):
    """
    Match original parquet rows to LCB questions.
    Returns: dict mapping row_idx -> question_idx
    """
    # Build question content -> index mapping
    question_to_idx = {}
    for i, instance in enumerate(lcb_dataset):
        question_to_idx[instance.question_content] = i
    
    row_to_question = {}
    
    for row_idx in range(len(original_df)):
        prompt = original_df[row_idx]['prompt'][0]
        
        # Extract prompt text
        if isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], dict):
                prompt_text = prompt[0].get('content', '')
            else:
                prompt_text = str(prompt[0])
        else:
            prompt_text = str(prompt)
        
        # Find matching question
        for q_content, q_idx in question_to_idx.items():
            if q_content in prompt_text:
                row_to_question[row_idx] = q_idx
                break
    
    print(f"  Matched {len(row_to_question)}/{len(original_df)} rows to LCB questions")
    return row_to_question


def evaluate_certainty_correctness(
    certainty_path,
    lcb_version,
    num_workers=16,
    timeout=120,
    save=False,
):
    """
    Evaluate LCB correctness for certainty parquet.
    
    Returns:
        updated_df: DataFrame with is_correct updated
        metrics: dict with pass@k metrics
        per_response_results: list of (row_idx, resp_idx, is_correct, test_results)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {Path(certainty_path).name}")
    print(f"LCB Version: {lcb_version}")
    print(f"{'='*60}")
    
    # Load data
    certainty_df, original_df = load_certainty_and_original(certainty_path)
    
    # Check if already evaluated
    lcb_mask = certainty_df['is_correct'] == -1.0
    lcb_count = lcb_mask.sum()
    
    if lcb_count == 0:
        print("  All samples already evaluated, skipping...")
        # Compute metrics from existing results
        # Group by question for pass@k
        return certainty_df, None, []
    
    print(f"  LCB samples to evaluate: {lcb_count}")
    
    # Load LCB dataset
    version_config = LCB_VERSION_CONFIG[lcb_version]
    print(f"Loading LCB dataset ({lcb_version})...")
    lcb_dataset = load_code_generation_dataset(
        release_version=version_config["release_version"],
        start_date=version_config["start_date"],
        end_date=version_config["end_date"]
    )
    eval_samples = [instance.get_evaluation_sample() for instance in lcb_dataset]
    print(f"  LCB questions: {len(lcb_dataset)}")
    
    # Match rows to questions
    row_to_question = match_to_lcb_dataset(original_df, lcb_dataset)
    
    # Build generations per question (same structure as codegen_metrics expects)
    # question_idx -> list of (df_idx, code)
    question_generations = defaultdict(list)
    df_idx_to_info = {}  # df_idx -> (question_idx, gen_idx_in_question)
    
    for df_idx, row in certainty_df.iterrows():
        if row['is_correct'] != -1.0:
            continue
        
        row_idx = row['row_idx']
        resp_idx = row['resp_idx']
        response = row['response']
        
        if row_idx not in row_to_question:
            continue
        
        q_idx = row_to_question[row_idx]
        code = process_response(response)
        
        gen_idx = len(question_generations[q_idx])
        question_generations[q_idx].append((df_idx, code))
        df_idx_to_info[df_idx] = (q_idx, gen_idx)
    
    # Prepare for evaluation
    # We need samples_list and generations_list aligned
    question_indices = sorted(question_generations.keys())
    samples_list = []
    generations_list = []
    question_idx_map = []  # maps position in list -> original question_idx
    
    for q_idx in question_indices:
        samples_list.append(eval_samples[q_idx])
        generations_list.append([code for _, code in question_generations[q_idx]])
        question_idx_map.append(q_idx)
    
    print(f"\nEvaluating {sum(len(g) for g in generations_list)} responses across {len(samples_list)} questions...")
    
    # Use the existing codegen_metrics function
    metrics_result = codegen_metrics(
        samples_list,
        generations_list,
        k_list=[1, 4, 8],
        num_process_evaluate=num_workers,
        timeout=timeout,
    )
    
    metrics = metrics_result[0]  # pass@k dict
    results = metrics_result[1]  # question_idx -> list of test_case_results per generation
    metadata = metrics_result[2]  # metadata
    
    print(f"\nPass@k Metrics:")
    print(f"  pass@1: {metrics.get('pass@1', 'N/A'):.4f}")
    print(f"  pass@4: {metrics.get('pass@4', 'N/A'):.4f}")
    print(f"  pass@8: {metrics.get('pass@8', 'N/A'):.4f}")
    
    # Update certainty_df with correctness
    per_response_results = []
    
    for list_idx, q_idx in enumerate(question_idx_map):
        gen_results = results[list_idx]  # list of test_case_results for each generation
        
        for gen_idx, (df_idx, _) in enumerate(question_generations[q_idx]):
            test_results = gen_results[gen_idx]
            
            # Check if all test cases passed
            is_correct = all(r == True or r == 1 for r in test_results) if test_results else False
            
            # Update DataFrame
            certainty_df.at[df_idx, 'is_correct'] = 1.0 if is_correct else 0.0
            
            row_idx = certainty_df.at[df_idx, 'row_idx']
            resp_idx = certainty_df.at[df_idx, 'resp_idx']
            per_response_results.append((row_idx, resp_idx, is_correct, test_results))
    
    # Summary
    evaluated = (certainty_df['is_correct'] >= 0).sum()
    correct = (certainty_df['is_correct'] == 1.0).sum()
    print(f"\nSummary:")
    print(f"  Total evaluated: {evaluated}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/evaluated*100:.2f}%" if evaluated > 0 else "N/A")
    
    # Save if requested
    if save:
        certainty_df.to_parquet(certainty_path)
        print(f"\nSaved updated parquet: {certainty_path}")
        
        # Also save metrics to CSV
        csv_path = str(certainty_path).replace('.parquet', '.pass.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'pass@1', 'pass@4', 'pass@8', 'accuracy', 'total', 'correct'])
            writer.writeheader()
            writer.writerow({
                'file': str(certainty_path),
                'pass@1': metrics.get('pass@1', ''),
                'pass@4': metrics.get('pass@4', ''),
                'pass@8': metrics.get('pass@8', ''),
                'accuracy': correct/evaluated if evaluated > 0 else 0,
                'total': evaluated,
                'correct': correct,
            })
        print(f"Saved metrics CSV: {csv_path}")
    
    return certainty_df, metrics, per_response_results


def batch_evaluate(base_dir, lcb_version='auto', num_workers=16, timeout=120, save=False):
    """Batch evaluate all LCB certainty files under base_dir"""
    base_dir = Path(base_dir)
    
    # Find all LCB certainty files
    patterns = ['**/livecodebench_v5_certainty.parquet', '**/livecodebench_v6_certainty.parquet']
    files = []
    for pattern in patterns:
        files.extend(base_dir.glob(pattern))
    files = sorted(set(files))
    
    print(f"Found {len(files)} LCB certainty files")
    
    all_metrics = []
    
    for f in files:
        # Detect version from filename
        if lcb_version == 'auto':
            if 'v6' in f.name:
                version = 'v6'
            else:
                version = 'v5'
        else:
            version = lcb_version
        
        try:
            _, metrics, _ = evaluate_certainty_correctness(
                f, version, num_workers, timeout, save
            )
            
            if metrics:
                all_metrics.append({
                    'file': str(f),
                    'version': version,
                    **metrics
                })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    if all_metrics:
        print(f"\n{'='*60}")
        print("Summary of All Evaluations")
        print(f"{'='*60}")
        print(f"{'File':<50} {'pass@1':>8} {'pass@4':>8} {'pass@8':>8}")
        print("-" * 76)
        for m in all_metrics:
            fname = Path(m['file']).name
            print(f"{fname:<50} {m.get('pass@1', 0):>8.4f} {m.get('pass@4', 0):>8.4f} {m.get('pass@8', 0):>8.4f}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Compute LCB correctness for certainty parquet')
    parser.add_argument('--certainty_file', type=str, help='Single certainty parquet file')
    parser.add_argument('--base_dir', type=str, help='Base directory for batch processing')
    parser.add_argument('--batch', action='store_true', help='Batch process all files')
    parser.add_argument('--lcb_version', type=str, default='auto', choices=['v5', 'v6', 'auto'],
                       help='LCB version (auto: detect from filename)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per test case')
    parser.add_argument('--save', action='store_true', help='Save updated parquet')
    args = parser.parse_args()
    
    if args.batch or args.base_dir:
        base_dir = args.base_dir or '/data4/user/jin509/Archer_eval/output/ArcherCodeR'
        batch_evaluate(base_dir, args.lcb_version, args.num_workers, args.timeout, args.save)
    elif args.certainty_file:
        version = args.lcb_version
        if version == 'auto':
            version = 'v6' if 'v6' in args.certainty_file else 'v5'
        evaluate_certainty_correctness(
            args.certainty_file, version, args.num_workers, args.timeout, args.save
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

