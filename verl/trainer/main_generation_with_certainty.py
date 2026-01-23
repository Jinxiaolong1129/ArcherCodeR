#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    :   2025/01/15
@Author  :   Based on main_generation.py, extended for certainty analysis
@File    :   main_generation_with_certainty.py

This script extends main_generation.py to compute certainty metrics
(self_certainty, trajectory_entropy, token_entropy, prob_disparity)
for analyzing model calibration (True Certain, True Uncertain, False Certain, False Uncertain).

Key difference from main_generation.py:
- Uses role="actor_rollout" instead of role="rollout" to enable both vLLM generation and FSDP compute_log_prob
- After vLLM generation, calls compute_log_prob to get certainty metrics
- Classifies samples into TC/TU/FC/FU categories
"""

import csv
import ray
import numpy as np
import hydra
import os
import json
import torch
from tabulate import tabulate

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint
from collections import defaultdict

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from concurrent.futures import ProcessPoolExecutor, as_completed


def parallel_compute_score(evaluation_func, response_str, ground_truth, data_sources, max_workers=64):
    """Parallel reward computation."""
    with tqdm(total=len(response_str), desc="Computing rewards") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluation_func, data_sources[index], response_str[index], ground_truth[index]): index
                for index in range(len(response_str))
            }
            results = {}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error computing reward for index {index}: {e}")
                    results[index] = 0.0
                pbar.update(1)

    return [results[i] for i in range(len(response_str))]


def compute_masked_mean(values, mask):
    """Compute masked mean along the last dimension."""
    masked_values = values * mask
    return masked_values.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)


def classify_sample(is_correct, certainty_value, threshold, high_means_certain=True):
    """
    Classify sample into TC/TU/FC/FU categories.
    
    Args:
        is_correct: Whether the response is correct
        certainty_value: The certainty metric value
        threshold: The threshold for classifying as "certain"
        high_means_certain: If True, high values mean certain (e.g., self_certainty, prob_disparity)
                           If False, high values mean uncertain (e.g., entropy)
    
    Returns:
        Category string: 'TC', 'TU', 'FC', or 'FU'
    """
    if high_means_certain:
        is_certain = certainty_value >= threshold
    else:
        is_certain = certainty_value <= threshold
    
    if is_correct and is_certain:
        return 'TC'  # True Certain
    elif is_correct and not is_certain:
        return 'TU'  # True Uncertain
    elif not is_correct and is_certain:
        return 'FC'  # False Certain
    else:
        return 'FU'  # False Uncertain


@hydra.main(config_path="config", config_name="generation_with_certainty", version_base=None)
def main(config):
    run_generation_with_certainty(config)


def run_generation_with_certainty(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Read dataset
    try:
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [chat.tolist() if hasattr(chat, 'tolist') else chat for chat in chat_lst]
    except Exception as e:
        import json
        config.data.path = config.data.path.replace('.parquet', '.json')
        with open(config.data.path, 'r') as f:
            dataset = pd.read_json(f)
        chat_lst = dataset[config.data.prompt_key].tolist()

    print(f'Original dataset len: {len(dataset)}')

    # Filter out too long prompts
    prompt_key = "prompt"
    dataset = dataset[dataset.apply(lambda doc: len(
        tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= config.rollout.prompt_length, axis=1)]
    
    # Re-extract chat_lst after filtering
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() if hasattr(chat, 'tolist') else chat for chat in chat_lst]

    print(f'Filtered dataset len: {len(dataset)}')

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ KEY CHANGE: Use "actor_rollout" instead of "rollout" ============
    # This enables both vLLM generation AND FSDP compute_log_prob
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker), 
        config=config, 
        role="actor_rollout"  # Changed from "rollout" to enable compute_log_prob
    )
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size
    
    # Storage for all results
    all_results = []
    response_length = config.rollout.response_length

    for batch_idx in range(num_batch):
        print(f'\n[{batch_idx+1}/{num_batch}] Processing batch...')
        batch_start = batch_idx * config_batch_size
        batch_end = min((batch_idx + 1) * config_batch_size, total_samples)
        batch_chat_lst = chat_lst[batch_start:batch_end]
        
        if len(batch_chat_lst) == 0:
            continue

        # Repeat for n_samples
        repeated_chat_lst = []
        for chat in batch_chat_lst:
            repeated_chat_lst.extend([chat] * config.data.n_samples)

        inputs = tokenizer.apply_chat_template(
            repeated_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors='pt',
            return_dict=True,
            tokenize=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'position_ids': position_ids
        }
        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]

        # Pad if needed for DP
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(f'  Added {dummy_data_size} dummy samples for DP alignment')

        # ============ Stage 1: vLLM Generation ============
        print(f'  [Stage 1] Generating with vLLM...')
        gen_output = wg.generate_sequences(data)
        
        # ============ Stage 2: FSDP compute_log_prob ============
        print(f'  [Stage 2] Computing certainty metrics with FSDP...')
        
        # Prepare full data (prompt + response) for compute_log_prob
        # gen_output contains: input_ids (full), attention_mask, responses, etc.
        full_input_ids = gen_output.batch['input_ids']  # This is prompt + response
        full_attention_mask = gen_output.batch['attention_mask']
        responses = gen_output.batch['responses']
        
        # Recompute position_ids for full sequence
        full_position_ids = compute_position_id_with_mask(full_attention_mask)
        
        # Create data for compute_log_prob
        compute_data = DataProto.from_dict({
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'position_ids': full_position_ids,
            'responses': responses,
        })
        
        # Call compute_log_prob to get certainty metrics
        log_prob_output = wg.compute_log_prob(compute_data)
        
        # Remove dummy data
        gen_output = gen_output[:real_batch_size]
        log_prob_output = log_prob_output[:real_batch_size]

        # ============ Extract metrics ============
        old_log_probs = log_prob_output.batch['old_log_probs']      # (batch, response_len)
        entropys = log_prob_output.batch['entropys']                # (batch, response_len)
        self_certaintys = log_prob_output.batch['self_certaintys']  # (batch, response_len)
        prob_disparitys = log_prob_output.batch['prob_disparitys']  # (batch, response_len)
        
        # Get response mask
        responses = gen_output.batch['responses'][:real_batch_size]
        response_attention = gen_output.batch['attention_mask'][:real_batch_size, -response_length:]
        response_mask = response_attention.float()
        
        # Compute sentence-level metrics
        sentence_self_certainty = compute_masked_mean(self_certaintys, response_mask)
        sentence_trajectory_entropy = -compute_masked_mean(old_log_probs, response_mask)  # Negative log prob
        sentence_token_entropy = compute_masked_mean(entropys, response_mask)
        sentence_prob_disparity = compute_masked_mean(prob_disparitys, response_mask)

        # Decode responses
        response_texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        # Remove padding from response texts
        pad_token = tokenizer.pad_token
        response_texts = [text.replace(pad_token, '').strip() for text in response_texts]

        # Store results
        for i in range(real_batch_size):
            all_results.append({
                'batch_idx': batch_idx,
                'sample_idx': batch_start + (i // config.data.n_samples),
                'response_idx': i % config.data.n_samples,
                'response': response_texts[i],
                'self_certainty': sentence_self_certainty[i].item(),
                'trajectory_entropy': sentence_trajectory_entropy[i].item(),
                'token_entropy': sentence_token_entropy[i].item(),
                'prob_disparity': sentence_prob_disparity[i].item(),
            })

        print(f'  Processed {real_batch_size} samples')

    print(f'\n{"="*60}')
    print(f'Generation complete. Total samples: {len(all_results)}')
    print(f'{"="*60}')

    # ============ Compute rewards (correctness) ============
    print('\nComputing rewards...')
    
    reward_model_data = dataset[config.data.reward_model_key].tolist()
    data_sources = dataset[config.data.data_source_key].tolist()
    
    # Flatten for parallel computation
    response_strs = [r['response'] for r in all_results]
    ground_truths = []
    data_sources_flat = []
    
    for result in all_results:
        sample_idx = result['sample_idx']
        ground_truths.append(reward_model_data[sample_idx]['ground_truth'])
        data_sources_flat.append(data_sources[sample_idx])

    try:
        from rewards.general_reward import general_reward_fn
        scores = parallel_compute_score(
            general_reward_fn,
            response_strs,
            ground_truths,
            data_sources_flat,
            max_workers=config.data.get('reward_workers', 64)
        )
    except Exception as e:
        print(f"Error in reward computation: {e}")
        print("Setting all scores to 0")
        scores = [0.0] * len(response_strs)

    # Add scores to results
    for i, result in enumerate(all_results):
        result['reward'] = float(scores[i])
        result['is_correct'] = scores[i] > 0
        result['ground_truth'] = ground_truths[i]
        result['data_source'] = data_sources_flat[i]

    # ============ Classify into TC/TU/FC/FU ============
    print('\nClassifying samples...')
    
    # Get the training method from config to determine which certainty metric to use
    training_method = config.get('analysis', {}).get('training_method', 'INTUITOR')
    
    # Mapping: metric_key, high_means_certain
    metric_map = {
        'INTUITOR': ('self_certainty', True),
        'TRAJECTORY_ENTROPY': ('trajectory_entropy', False),  # High entropy = uncertain
        'TOKEN_ENTROPY': ('token_entropy', False),            # High entropy = uncertain
        'PROB_DISPARITY': ('prob_disparity', True),           # High disparity = certain
    }
    
    certainty_key, high_means_certain = metric_map.get(training_method, ('self_certainty', True))
    
    # Compute threshold (median)
    all_certainty = [r[certainty_key] for r in all_results]
    threshold = np.median(all_certainty)
    
    print(f'  Training method: {training_method}')
    print(f'  Using certainty metric: {certainty_key}')
    print(f'  Threshold (median): {threshold:.4f}')
    
    # Classify each sample
    for result in all_results:
        result['category'] = classify_sample(
            result['is_correct'],
            result[certainty_key],
            threshold,
            high_means_certain
        )

    # ============ Statistics ============
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    
    # Overall accuracy
    accuracy = np.mean([r['is_correct'] for r in all_results])
    print(f'\nOverall Accuracy: {accuracy*100:.2f}%')
    
    # Category distribution
    category_counts = defaultdict(int)
    for r in all_results:
        category_counts[r['category']] += 1
    
    total = len(all_results)
    print(f'\nCategory Distribution (N={total}):')
    for cat in ['TC', 'TU', 'FC', 'FU']:
        count = category_counts[cat]
        pct = 100 * count / total
        print(f'  {cat}: {count:5d} ({pct:5.1f}%)')
    
    # Key metrics
    tc_count = category_counts['TC']
    fc_count = category_counts['FC']
    overconfidence_rate = fc_count / (tc_count + fc_count + 1e-8) * 100
    print(f'\nOverconfidence Rate (FC/(TC+FC)): {overconfidence_rate:.1f}%')
    
    # Certainty statistics
    print(f'\nCertainty Statistics ({certainty_key}):')
    print(f'  Mean: {np.mean(all_certainty):.4f}')
    print(f'  Std:  {np.std(all_certainty):.4f}')
    print(f'  Min:  {np.min(all_certainty):.4f}')
    print(f'  Max:  {np.max(all_certainty):.4f}')
    
    # Certainty by correctness
    correct_certainty = [r[certainty_key] for r in all_results if r['is_correct']]
    incorrect_certainty = [r[certainty_key] for r in all_results if not r['is_correct']]
    
    if correct_certainty:
        print(f'\n  Correct samples mean {certainty_key}: {np.mean(correct_certainty):.4f}')
    if incorrect_certainty:
        print(f'  Incorrect samples mean {certainty_key}: {np.mean(incorrect_certainty):.4f}')

    # ============ Save results ============
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    
    # Save as parquet
    results_df = pd.DataFrame(all_results)
    output_path = config.data.output_path.replace('.parquet', '_with_certainty.parquet')
    results_df.to_parquet(output_path)
    print(f'\nSaved results to: {output_path}')
    
    # Save summary as JSON
    summary = {
        'model_path': config.model.path,
        'dataset': os.path.basename(config.data.path),
        'training_method': training_method,
        'certainty_metric': certainty_key,
        'threshold': float(threshold),
        'total_samples': total,
        'accuracy': float(accuracy),
        'category_counts': dict(category_counts),
        'overconfidence_rate': float(overconfidence_rate),
        'certainty_stats': {
            'mean': float(np.mean(all_certainty)),
            'std': float(np.std(all_certainty)),
            'min': float(np.min(all_certainty)),
            'max': float(np.max(all_certainty)),
        }
    }
    
    summary_path = output_path.replace('.parquet', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary to: {summary_path}')
    
    # Save pass@k metrics (compatible with original script)
    n_samples = config.data.n_samples
    if n_samples > 1:
        # Reshape scores for pass@k computation
        num_prompts = len(all_results) // n_samples
        scores_reshaped = np.array([r['reward'] for r in all_results]).reshape(num_prompts, n_samples)
        pass_at_n = np.mean(np.max(scores_reshaped, axis=-1))
        pass_at_1 = np.mean(scores_reshaped)
        
        print(f'\nPass@1: {pass_at_1*100:.2f}%')
        print(f'Pass@{n_samples}: {pass_at_n*100:.2f}%')
        
        csv_path = config.data.output_path + '.pass.csv'
        row_data = {
            'model_path': config.model.path,
            'dataset': os.path.basename(config.data.path),
            'pass@1': pass_at_1,
            f'pass@{n_samples}': pass_at_n,
            'accuracy': accuracy,
            'overconfidence_rate': overconfidence_rate,
        }
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    print('\n' + '='*60)
    print('DONE')
    print('='*60)


if __name__ == "__main__":
    main()

