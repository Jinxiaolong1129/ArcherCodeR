#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute certainty metrics (self_certainty, entropy, prob_disparity) for existing responses.
Skips vLLM generation and only uses FSDP Actor for forward pass.
"""

import csv
import ray
import numpy as np
import hydra
import os
import torch
from tabulate import tabulate

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint
from tqdm import tqdm

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def parallel_compute_score(evaluation_func, response_str, ground_truth, data_sources, max_workers=64):
    with tqdm(total=len(response_str), desc="Computing scores") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluation_func, data_sources[index], response_str[index], ground_truth[index]): index
                for index in range(len(response_str))
            }
            results = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return [results[i] for i in range(len(response_str))]


@hydra.main(config_path="config", config_name="compute_certainty", version_base=None)
def main(config):
    run_compute_certainty(config)


def run_compute_certainty(config) -> None:
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

    # Load existing parquet with responses
    input_path = config.data.input_path
    print(f"Loading existing responses from: {input_path}")
    
    # Try different methods to read parquet (some files have nested structures that fail)
    try:
        dataset = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Standard parquet read failed: {e}")
        print("Trying polars...")
        try:
            import polars as pl
            pl_df = pl.read_parquet(input_path)
            dataset = pl_df.to_pandas()
        except ImportError:
            print("polars not installed, trying pyarrow fallback...")
            import pyarrow.parquet as pq
            import pyarrow as pa
            # Read columns that work with pyarrow, skip problematic ones
            with pa.memory_map(input_path, 'r') as source:
                pf = pq.ParquetFile(source)
                all_cols = pf.schema_arrow.names
                # Try reading all at once first
                data_dict = {}
                for col_name in all_cols:
                    try:
                        col_table = pf.read(columns=[col_name])
                        data_dict[col_name] = col_table.to_pandas()[col_name]
                    except:
                        # Skip problematic columns (like nested structs)
                        print(f"  Skipping column {col_name} (nested structure)")
                dataset = pd.DataFrame(data_dict)
    print(f"Dataset loaded: {len(dataset)} rows")
    
    # Check required columns
    assert 'responses' in dataset.columns, "Input parquet must contain 'responses' column"
    assert config.data.prompt_key in dataset.columns, f"Input parquet must contain '{config.data.prompt_key}' column"
    
    # Load tokenizer
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize FSDP Actor only (no vLLM rollout)
    # role="actor" only initializes the FSDP model for compute_log_prob
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker), 
        config=config, 
        role="actor"  # Only Actor, no rollout
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    
    # Get data parallel size
    dp_size = wg.world_size
    
    # Prepare data
    prompts = dataset[config.data.prompt_key].tolist()
    responses_list = dataset['responses'].tolist()
    
    # Flatten prompts and responses for batch processing
    all_prompts = []
    all_responses = []
    all_indices = []  # To track original row and response index
    
    for row_idx, (prompt, responses) in enumerate(zip(prompts, responses_list)):
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
        for resp_idx, response in enumerate(responses):
            all_prompts.append(prompt_list)
            all_responses.append(response)
            all_indices.append((row_idx, resp_idx))
    
    total_samples = len(all_prompts)
    print(f"Total samples to process: {total_samples}")
    
    # Process in batches
    config_batch_size = config.data.batch_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size
    
    all_self_certaintys = []
    all_entropys = []
    all_prob_disparitys = []
    all_log_probs = []
    
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config_batch_size
        end_idx = min((batch_idx + 1) * config_batch_size, total_samples)
        
        batch_prompts = all_prompts[start_idx:end_idx]
        batch_responses = all_responses[start_idx:end_idx]
        
        print(f'[{batch_idx+1}/{num_batch}] Processing {len(batch_prompts)} samples...')
        
        # Tokenize prompts with chat template
        prompt_inputs = tokenizer.apply_chat_template(
            batch_prompts,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors='pt',
            return_dict=True,
            tokenize=True
        )
        
        # Tokenize responses
        response_inputs = tokenizer(
            batch_responses,
            padding=True,
            truncation=True,
            max_length=config.rollout.response_length,
            return_tensors='pt',
            add_special_tokens=False  # Don't add BOS for responses
        )
        
        # Concatenate prompt and response
        prompt_ids = prompt_inputs['input_ids']
        response_ids = response_inputs['input_ids']
        
        # Create full input_ids by concatenating prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Create attention mask
        prompt_mask = prompt_inputs['attention_mask']
        response_mask = response_inputs['attention_mask']
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        
        # Create position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Create responses tensor (just the response part)
        responses = response_ids.clone()
        
        batch_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'responses': responses
        }
        
        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        
        # Pad to dp_size if needed
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(f'Added {dummy_data_size} dummy samples for dp_size alignment')
        
        # Compute log_prob and certainty metrics
        print(f'[{batch_idx+1}/{num_batch}] Computing certainty metrics...')
        output = wg.compute_log_prob(data)
        
        # Remove dummy data
        output = output[:real_batch_size]
        
        # Extract metrics (shape: [batch_size, response_length])
        self_certaintys = output.batch['self_certaintys'].numpy()
        entropys = output.batch['entropys'].numpy()
        prob_disparitys = output.batch['prob_disparitys'].numpy()
        old_log_probs = output.batch['old_log_probs'].numpy()
        
        # Get response mask for this batch
        # The output has shape [batch_size, responses.shape[-1]]
        # We need to use the response_inputs attention mask
        batch_response_mask = response_mask[:real_batch_size].numpy()
        
        # The output length matches response tensor length
        output_len = self_certaintys.shape[1]
        mask_len = batch_response_mask.shape[1]
        
        # Compute sentence-level mean metrics
        for i in range(real_batch_size):
            # Truncate or pad mask to match output length
            if mask_len >= output_len:
                mask = batch_response_mask[i][:output_len].astype(bool)
            else:
                mask = np.zeros(output_len, dtype=bool)
                mask[:mask_len] = batch_response_mask[i].astype(bool)
            
            # Sentence-level mean (only for valid response tokens)
            if mask.sum() > 0:
                all_self_certaintys.append(float(self_certaintys[i][mask].mean()))
                all_entropys.append(float(entropys[i][mask].mean()))
                all_prob_disparitys.append(float(prob_disparitys[i][mask].mean()))
                all_log_probs.append(float(old_log_probs[i][mask].mean()))
            else:
                # If no valid tokens, use mean of all non-zero values
                valid_mask = self_certaintys[i] != 0
                if valid_mask.sum() > 0:
                    all_self_certaintys.append(float(self_certaintys[i][valid_mask].mean()))
                    all_entropys.append(float(entropys[i][valid_mask].mean()))
                    all_prob_disparitys.append(float(prob_disparitys[i][valid_mask].mean()))
                    all_log_probs.append(float(old_log_probs[i][valid_mask].mean()))
                else:
                    all_self_certaintys.append(0.0)
                    all_entropys.append(0.0)
                    all_prob_disparitys.append(0.0)
                    all_log_probs.append(0.0)
    
    print(f"Processed {len(all_self_certaintys)} total responses")
    
    # Compute correctness scores
    data_sources = dataset[config.data.data_source_key].tolist()
    reward_model_data = dataset[config.data.reward_model_key].tolist()
    
    # Expand data_sources and ground_truth to match flattened responses
    all_data_sources = []
    all_ground_truths = []
    for row_idx, resp_idx in all_indices:
        all_data_sources.append(data_sources[row_idx])
        all_ground_truths.append(reward_model_data[row_idx]['ground_truth'])
    
    print("Computing correctness scores...")
    try:
        from rewards.general_reward import general_reward_fn
        all_scores = parallel_compute_score(
            general_reward_fn,
            all_responses,
            all_ground_truths,
            all_data_sources,
        )
    except Exception as e:
        print(f"Error computing scores: {e}. Setting all to 0.")
        all_scores = [0.0] * len(all_responses)
    
    # Convert boolean scores to float
    all_scores = [1.0 if s else 0.0 for s in all_scores]
    
    # Create results dataframe
    results = []
    for i, (row_idx, resp_idx) in enumerate(all_indices):
        results.append({
            'row_idx': row_idx,
            'resp_idx': resp_idx,
            'response': all_responses[i],
            'is_correct': all_scores[i],
            'self_certainty': all_self_certaintys[i],
            'entropy': all_entropys[i],
            'prob_disparity': all_prob_disparitys[i],
            'mean_log_prob': all_log_probs[i],
            'trajectory_entropy': -all_log_probs[i],  # trajectory_entropy = -mean_log_prob
            'data_source': all_data_sources[i],
            'ground_truth': all_ground_truths[i],
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = config.data.output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        makedirs(output_dir, exist_ok=True)
    
    results_df.to_parquet(output_path)
    print(f"Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    correct_df = results_df[results_df['is_correct'] == 1.0]
    incorrect_df = results_df[results_df['is_correct'] == 0.0]
    
    summary_data = [
        ['Total samples', len(results_df)],
        ['Correct answers', len(correct_df)],
        ['Incorrect answers', len(incorrect_df)],
        ['Accuracy', f"{len(correct_df) / len(results_df) * 100:.2f}%"],
        ['', ''],
        ['Metric', 'Correct Mean | Incorrect Mean'],
        ['self_certainty', f"{correct_df['self_certainty'].mean():.4f} | {incorrect_df['self_certainty'].mean():.4f}"],
        ['entropy', f"{correct_df['entropy'].mean():.4f} | {incorrect_df['entropy'].mean():.4f}"],
        ['prob_disparity', f"{correct_df['prob_disparity'].mean():.4f} | {incorrect_df['prob_disparity'].mean():.4f}"],
        ['trajectory_entropy', f"{correct_df['trajectory_entropy'].mean():.4f} | {incorrect_df['trajectory_entropy'].mean():.4f}"],
    ]
    
    print(tabulate(summary_data, tablefmt='grid'))
    
    # Save summary to JSON
    summary_json_path = output_path.replace('.parquet', '.summary.json')
    import json
    summary = {
        'total_samples': len(results_df),
        'correct_answers': len(correct_df),
        'incorrect_answers': len(incorrect_df),
        'accuracy': len(correct_df) / len(results_df),
        'metrics': {
            'self_certainty': {
                'correct_mean': float(correct_df['self_certainty'].mean()),
                'incorrect_mean': float(incorrect_df['self_certainty'].mean()),
            },
            'entropy': {
                'correct_mean': float(correct_df['entropy'].mean()),
                'incorrect_mean': float(incorrect_df['entropy'].mean()),
            },
            'prob_disparity': {
                'correct_mean': float(correct_df['prob_disparity'].mean()),
                'incorrect_mean': float(incorrect_df['prob_disparity'].mean()),
            },
            'trajectory_entropy': {
                'correct_mean': float(correct_df['trajectory_entropy'].mean()),
                'incorrect_mean': float(incorrect_df['trajectory_entropy'].mean()),
            },
        }
    }
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_json_path}")


if __name__ == "__main__":
    main()

