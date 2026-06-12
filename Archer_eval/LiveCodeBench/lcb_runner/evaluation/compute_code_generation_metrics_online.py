#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:47:46
@Author  :   wangjiakang
@File    :   compute_code_generation_metrics.py
'''

# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import os
import sys

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import base64
import zlib
import pickle
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

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
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list[0]


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
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
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

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)}...")

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

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(
            generations_list[0]
        ), f"{len(final_metadata[i])=}"

    return [metrics, results, final_metadata]

import argparse
import csv
import re
import wandb
import polars as pl
import pandas as pd

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

THOUGHT_DELIMITER_END = "</think>"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval_file",
    required=True,
    type=str,
)
parser.add_argument(
    "--project_name",
    required=True,
    type=str,
)
parser.add_argument(
    "--experiment_name",
    required=True,
    type=str,
)
parser.add_argument(
    "--global_step",
    required=True,
    type=str,
)
parser.add_argument(
    "--lcb_version",
    default="v5",
    type=str,
    choices=["v5", "v6"],
    help="LiveCodeBench version: v5 (2024-08-01 ~ 2025-02-01) or v6 (2025-02-01 ~)"
)
parser.add_argument(
    "--num_workers",
    default=128,
    type=int,
    help="Number of parallel CPU workers for evaluation (default: 128)"
)
parser.add_argument(
    "--benchmark_json",
    default=None,
    type=str,
    help="Local benchmark json file path (preferred; skips HF download).",
)
args = parser.parse_args()

# 版本配置
# v5: test1-test5.jsonl (2024-08-01 ~ 2025-02-01)
# v6: test6.jsonl only (2025-02-01 ~ 最新)
LCB_VERSION_CONFIG = {
    "v5": {"release_version": "release_v5", "start_date": "2024-08-01", "end_date": "2025-02-01"},
    "v6": {"release_version": "release_v6", "start_date": "2025-02-01", "end_date": None},
}


def extract_answer(model_solution):
    # pattern = r"```(?:\w+)?\n(.*?)\n```"
    pattern = r"```python\n(.*?)```"
    match = re.findall(pattern, model_solution, re.DOTALL)
    if len(match) == 0:
        return None
    else:
        code = match[-1]
        return code


def _parse_json_if_needed(obj):
    if isinstance(obj, str):
        return json.loads(obj)
    return obj


def _normalize_test_list(obj):
    """Return a flat list of {'input','output'} test dicts."""
    try:
        obj = _parse_json_if_needed(obj)
    except Exception:
        # Some v6 private_test_cases are compressed/base64-encoded payloads.
        if isinstance(obj, str):
            try:
                obj = pickle.loads(zlib.decompress(base64.b64decode(obj.encode("utf-8"))))
                obj = _parse_json_if_needed(obj)
            except Exception:
                return []
        else:
            return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _extract_prompt_text(row):
    prompt_val = row.get("prompt")
    # Common format: [{'role':'user','content':'...'}]
    if isinstance(prompt_val, list) and prompt_val:
        first = prompt_val[0]
        if isinstance(first, dict) and "content" in first:
            return first["content"]
        if isinstance(first, str):
            return first
    # Fallbacks for other local dumps
    for key in ("question_content", "question", "instruction"):
        if isinstance(row.get(key), str):
            return row[key]
    return None


def _extract_tests_from_ground_truth(ground_truth_raw):
    """
    Support both formats:
    - v5 style: ground_truth is JSON list of test dicts
    - v6 style: ground_truth is JSON object with public/private_test_cases
    """
    gt = _parse_json_if_needed(ground_truth_raw)

    # v5-style direct list
    if isinstance(gt, list):
        return _normalize_test_list(gt)

    # v6-style object
    if isinstance(gt, dict):
        tests = []
        for k in ("public_test_cases", "private_test_cases"):
            if k in gt:
                tests.extend(_normalize_test_list(gt[k]))
        return tests

    return []


def build_eval_samples_from_local_benchmark(benchmark_json_path):
    print(f"Loading local benchmark json: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt_to_eval_sample = {}
    bad_rows = 0
    for row in data:
        prompt_text = _extract_prompt_text(row)
        if not isinstance(prompt_text, str) or len(prompt_text) == 0:
            bad_rows += 1
            continue

        reward_model = row.get("reward_model", {})
        ground_truth_raw = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if ground_truth_raw is None:
            bad_rows += 1
            continue
        try:
            tests = _extract_tests_from_ground_truth(ground_truth_raw)
        except Exception:
            bad_rows += 1
            continue
        if not isinstance(tests, list) or len(tests) == 0:
            bad_rows += 1
            continue

        inputs = [t.get("input", "") for t in tests]
        outputs = [t.get("output", "") for t in tests]
        eval_sample = {
            "input_output": json.dumps(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fn_name": None,
                }
            )
        }
        prompt_to_eval_sample[prompt_text] = eval_sample

    print(f"Loaded local benchmark prompts: {len(prompt_to_eval_sample)}, skipped bad rows: {bad_rows}")
    return prompt_to_eval_sample


if __name__ == "__main__":

    parquet_file = args.eval_file
    csv_path = parquet_file + '.pass.lcb.csv'
    print(f"Evaluating {parquet_file}...")
    print(f"CSV path: {csv_path}")
    
    if not os.path.exists(csv_path):
        dataframe = pl.read_parquet(parquet_file)

        prompt2responses = {}
        for i in range(len(dataframe)):
            prompt = dataframe[i]['prompt'][0][0]['content']
            responses = []
            for response in dataframe[i]['responses'][0]:
                if THOUGHT_DELIMITER_END not in response:
                    model_solution = response
                    solution_code = extract_answer(model_solution)
                    if solution_code is None or not isinstance(solution_code, str):
                        solution_code = ''
                else:
                    model_solution = response.split(THOUGHT_DELIMITER_END)[1]
                    solution_code = extract_answer(model_solution)
                    if solution_code is None or not isinstance(solution_code, str):
                        solution_code = ''
                responses.append(solution_code)
            prompt2responses[prompt] = responses

        eval_samples = []
        generations = []
        matched_count = 0

        if args.benchmark_json:
            prompt_to_eval_sample = build_eval_samples_from_local_benchmark(args.benchmark_json)
            unmatched_prompts = []
            for bench_prompt, eval_sample in prompt_to_eval_sample.items():
                responses = prompt2responses.get(bench_prompt)
                if responses is None:
                    # Fallback fuzzy match for prompt template variations.
                    for p, r in prompt2responses.items():
                        if bench_prompt in p or p in bench_prompt:
                            responses = r
                            break
                if responses is None:
                    unmatched_prompts.append(bench_prompt[:120])
                    continue
                eval_samples.append(eval_sample)
                generations.append(responses)
                matched_count += 1
            print(f"Matched {matched_count}/{len(prompt_to_eval_sample)} local benchmark prompts")
            if unmatched_prompts:
                print(f"Unmatched local prompts: {unmatched_prompts[:5]}...")
        else:
            version_config = LCB_VERSION_CONFIG[args.lcb_version]
            print(f"Using LCB version: {args.lcb_version}, config: {version_config}")
            lcb_dataset = load_code_generation_dataset(
                release_version=version_config["release_version"],
                start_date=version_config["start_date"],
                end_date=version_config["end_date"]
            )
            eval_samples = [instance.get_evaluation_sample() for instance in lcb_dataset]
            unmatched_questions = []
            for i in range(len(lcb_dataset)):
                found = False
                for prompt in prompt2responses:
                    if lcb_dataset[i].question_content in prompt:
                        generations.append(prompt2responses[prompt])
                        matched_count += 1
                        found = True
                        break
                if not found:
                    unmatched_questions.append(lcb_dataset[i].question_title)
            print(f"Matched {matched_count}/{len(lcb_dataset)} questions")
            if unmatched_questions:
                print(f"Unmatched questions: {unmatched_questions[:5]}...")

        if len(generations) == 0:
            raise RuntimeError("No matched benchmark samples found for this parquet file.")

        for responses in generations:
            assert len(responses) == len(generations[0]), (len(responses), len(generations[0]))

        metrics = codegen_metrics(
            eval_samples,
            generations,
            k_list=[1, 4, 8],
            num_process_evaluate=args.num_workers,
            timeout=120,
        )

        # Safely get pass@k metrics (may not exist if samples < k)
        pass_at_1 = metrics[0].get("pass@1", None)
        pass_at_4 = metrics[0].get("pass@4", None)
        pass_at_n = metrics[0].get("pass@8", None)
        
        print(f"Metrics available: {list(metrics[0].keys())}")
        print(f"pass@1: {pass_at_1}, pass@4: {pass_at_4}, pass@8: {pass_at_n}")

        # Save metrics to CSV
        dataset_name = os.path.basename(parquet_file)
        row_data = {
            'parquet_file': parquet_file,
            'dataset': dataset_name,
            'pass@1': pass_at_1,
            'pass@4': pass_at_4,
            'pass@8': pass_at_n
        }
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
    else:
        pass_score = pd.read_csv(csv_path)
        pass_at_1 = pass_score['pass@1'][0] if 'pass@1' in pass_score.columns else None
        pass_at_4 = pass_score['pass@4'][0] if 'pass@4' in pass_score.columns else None
        pass_at_n = pass_score['pass@8'][0] if 'pass@8' in pass_score.columns else None

    wandb.init(project=args.project_name, name=args.experiment_name, id=args.experiment_name, resume="allow", allow_val_change=True)
    wandb.define_metric("val/livecodebench/pass@1", step_metric="global_step")
    wandb.define_metric("val/livecodebench/pass@4", step_metric="global_step")
    wandb.define_metric("val/livecodebench/pass@8", step_metric="global_step")
    log_data = {"global_step": args.global_step}
    if pass_at_1 is not None:
        log_data["val/livecodebench/pass@1"] = pass_at_1
    if pass_at_4 is not None:
        log_data["val/livecodebench/pass@4"] = pass_at_4
    if pass_at_n is not None:
        log_data["val/livecodebench/pass@8"] = pass_at_n
    wandb.log(log_data)
    wandb.finish(exit_code=0)
