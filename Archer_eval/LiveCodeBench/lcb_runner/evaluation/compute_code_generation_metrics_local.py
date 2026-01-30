#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用本地完整版数据计算 LiveCodeBench 指标
支持 data/test/livecodebench_v5.json 和 livecodebench_v6.json

与 compute_code_generation_metrics_online.py 的区别：
- 使用本地 JSON 文件作为 test cases（完整版测试用例）
- 不依赖 HuggingFace 数据集

用法:
    python compute_code_generation_metrics_local.py \
        --eval_file /path/to/output.parquet \
        --testcase_file /path/to/livecodebench_v5.json \
        --project_name ArcherEval \
        --experiment_name MyExperiment \
        --global_step 100
"""

import os
import sys
import json
import argparse
import csv
import re
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import polars as pl
import pandas as pd

sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from local modules
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

THOUGHT_DELIMITER_END = "</think>"


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
        except Exception as e:
            if debug:
                print(f"Error in task {o_idx}: {e}")
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
            metadata.append(curr_metadata if 'curr_metadata' in dir() else {})
    return res, metadata


def codegen_metrics_local(
    eval_samples,
    generations_list,
    k_list=[1, 5],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    """计算代码生成指标"""
    
    samples_linear = []
    generations_linear = []
    remap_index = {}
    
    idx = 0
    for idx, (sample, generation_list) in enumerate(
        zip(eval_samples, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[idx]
        for generation in generation_list:
            samples_linear.append(sample)
            generations_linear.append(generation)
            remap_index[len(generations_linear) - 1] = idx
    
    print(f"Evaluating {len(generations_linear)}...")
    
    results_linear = {}
    metadatas_linear = {}
    
    results = defaultdict(list)
    metadatas = defaultdict(list)
    
    remainings = set(range(len(generations_linear)))
    
    with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
        futures = {
            executor.submit(
                evaluate_generations_by_problem,
                (
                    [generations_linear[idx]],
                    samples_linear[idx],
                    debug,
                    timeout,
                ),
            ): idx
            for idx in remainings
        }
        for future in tqdm(as_completed(futures), total=len(remainings)):
            idx = futures[future]
            results_linear[idx] = future.result()[0]
            metadatas_linear[idx] = future.result()[1]
    
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


def load_local_testcases(json_file: str) -> list[dict]:
    """从本地 JSON 文件加载测试用例"""
    print(f"Loading test cases from local file: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    eval_samples = []
    for item in data:
        # 从 reward_model.ground_truth 提取测试用例
        ground_truth = item.get("reward_model", {}).get("ground_truth", "[]")
        if isinstance(ground_truth, str):
            test_cases = json.loads(ground_truth)
        else:
            test_cases = ground_truth
        
        # 构建 eval_sample 格式（与 HF 数据格式兼容）
        inputs = [tc.get("input", "") for tc in test_cases]
        outputs = [tc.get("output", "") for tc in test_cases]
        
        # 获取 fn_name（如果有 metadata）
        fn_name = None
        if test_cases and "metadata" in test_cases[0]:
            fn_name = test_cases[0]["metadata"].get("func_name")
        
        eval_sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            }),
            "prompt": item.get("prompt", [{}])[0].get("content", ""),
        }
        eval_samples.append(eval_sample)
    
    print(f"Loaded {len(eval_samples)} problems from local file")
    return eval_samples, data


def extract_answer(model_solution):
    """提取 Python 代码块"""
    pattern = r"```python\n(.*?)```"
    match = re.findall(pattern, model_solution, re.DOTALL)
    if len(match) == 0:
        return None
    else:
        code = match[-1]
        return code


def main():
    parser = argparse.ArgumentParser(description="Evaluate code generation using local test cases")
    parser.add_argument("--eval_file", required=True, type=str, help="Path to parquet file with model outputs")
    parser.add_argument("--testcase_file", required=True, type=str, help="Path to local JSON file with test cases")
    parser.add_argument("--project_name", required=True, type=str, help="WandB project name")
    parser.add_argument("--experiment_name", required=True, type=str, help="Experiment name")
    parser.add_argument("--global_step", required=True, type=str, help="Global step")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for each test case")
    parser.add_argument("--num_workers", type=int, default=128, help="Number of parallel workers")
    args = parser.parse_args()

    parquet_file = args.eval_file
    csv_path = parquet_file + '.pass.lcb.local.csv'
    
    print(f"=" * 60)
    print(f"Evaluating with LOCAL test cases")
    print(f"=" * 60)
    print(f"Eval file: {parquet_file}")
    print(f"Testcase file: {args.testcase_file}")
    print(f"CSV path: {csv_path}")
    
    if not os.path.exists(csv_path):
        # 加载本地测试用例
        eval_samples, raw_data = load_local_testcases(args.testcase_file)
        
        # 读取模型输出
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
        
        # 匹配问题
        generations = []
        matched_count = 0
        unmatched = []
        
        for i, item in enumerate(raw_data):
            question_content = item.get("prompt", [{}])[0].get("content", "")
            found = False
            for prompt in prompt2responses:
                # 使用问题内容进行匹配
                if question_content in prompt or prompt in question_content:
                    generations.append(prompt2responses[prompt])
                    matched_count += 1
                    found = True
                    break
            if not found:
                # 尝试模糊匹配
                for prompt in prompt2responses:
                    # 提取核心问题内容（去掉格式指令）
                    if "### Format:" in question_content:
                        core_content = question_content.split("### Format:")[0].strip()
                    else:
                        core_content = question_content[:500]
                    
                    if core_content in prompt:
                        generations.append(prompt2responses[prompt])
                        matched_count += 1
                        found = True
                        break
                
                if not found:
                    unmatched.append(i)
                    generations.append([''] * len(list(prompt2responses.values())[0]))
        
        print(f"Matched {matched_count}/{len(raw_data)} questions")
        if unmatched:
            print(f"Unmatched indices: {unmatched[:10]}...")
        
        # 验证每个问题的响应数量一致
        for responses in generations:
            assert len(responses) == len(generations[0]), (len(responses), len(generations[0]))
        
        # 运行评估
        metrics = codegen_metrics_local(
            eval_samples,
            generations,
            k_list=[1, 4, 8],
            num_process_evaluate=args.num_workers,
            timeout=args.timeout,
        )
        
        pass_at_1 = metrics[0]["pass@1"]
        pass_at_4 = metrics[0]["pass@4"]
        pass_at_8 = metrics[0]["pass@8"]
        
        print(f"\n{'=' * 60}")
        print(f"Results (LOCAL test cases):")
        print(f"  pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
        print(f"  pass@4: {pass_at_4:.4f} ({pass_at_4*100:.2f}%)")
        print(f"  pass@8: {pass_at_8:.4f} ({pass_at_8*100:.2f}%)")
        print(f"{'=' * 60}")
        
        # 保存结果到 CSV
        dataset_name = os.path.basename(parquet_file)
        row_data = {
            'parquet_file': parquet_file,
            'testcase_file': args.testcase_file,
            'dataset': dataset_name,
            'pass@1': pass_at_1,
            'pass@4': pass_at_4,
            'pass@8': pass_at_8
        }
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            writer.writeheader()
            writer.writerow(row_data)
        
        print(f"Results saved to: {csv_path}")
    else:
        print(f"CSV already exists, loading...")
        pass_score = pd.read_csv(csv_path)
        pass_at_1 = pass_score['pass@1'][0]
        pass_at_4 = pass_score.get('pass@4', [None])[0]
        pass_at_8 = pass_score['pass@8'][0]
        
        print(f"  pass@1: {pass_at_1:.4f}")
        if pass_at_4:
            print(f"  pass@4: {pass_at_4:.4f}")
        print(f"  pass@8: {pass_at_8:.4f}")
    
    # 上传到 WandB
    try:
        import wandb
        wandb.init(
            project=args.project_name, 
            name=args.experiment_name, 
            id=args.experiment_name + "_local", 
            resume="allow", 
            allow_val_change=True
        )
        wandb.define_metric("val/livecodebench_local/pass@1", step_metric="global_step")
        wandb.define_metric("val/livecodebench_local/pass@4", step_metric="global_step")
        wandb.define_metric("val/livecodebench_local/pass@8", step_metric="global_step")
        
        log_data = {
            "val/livecodebench_local/pass@1": pass_at_1,
            "val/livecodebench_local/pass@8": pass_at_8,
            "global_step": int(args.global_step)
        }
        if pass_at_4:
            log_data["val/livecodebench_local/pass@4"] = pass_at_4
        
        wandb.log(log_data)
        wandb.finish(exit_code=0)
    except Exception as e:
        print(f"WandB logging failed: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()

