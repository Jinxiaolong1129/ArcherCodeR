#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:17:50
@Author  :   wangjiakang
@File    :   dapo_ray_trainer.py
'''


import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import time
import datetime
import logging
import pdb
import numpy as np
import math
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # 使用项目的tracking系统作为主要logger
        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            
            # 使用logger记录关键信息，pprint展示详细结构
            logger.info("Initial validation completed")
            print("📊 Initial validation metrics:")
            pprint(val_metrics)
            
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            
        print("🎯 Starting main training loop setup of RayDAPOTrainer...")
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        # 计算数据集信息
        train_dataset_size = len(self.train_dataloader.dataset) if hasattr(self.train_dataloader.dataset, '__len__') else "Unknown"
        train_dataloader_size = len(self.train_dataloader)
        actual_batch_size = self.config.data.train_batch_size
        
        print(f"🚀 Starting training loop of RayDAPOTrainer...")
        print(f"📊 Dataset info: {train_dataset_size} samples, {train_dataloader_size} batches")
        print(f"📦 Batch size: {actual_batch_size}, Total epochs: {self.config.trainer.total_epochs}")
        print(f"🎯 Total training steps: {self.total_training_steps}")
        
        # 用于追踪整体进度
        total_samples_processed = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            epoch_start_time = time.time()
            print(f"📚 ==================== Epoch {epoch + 1}/{self.config.trainer.total_epochs} ====================")
            print(f"📈 Epoch progress will process {train_dataloader_size} batches ({train_dataloader_size * actual_batch_size} samples)")
            
            # Epoch内的进度追踪
            epoch_samples_processed = 0
            batch_count = 0
            
            for batch_dict in self.train_dataloader:
                batch_count += 1
                step_start_time = time.time()
                
                # 详细的进度信息
                epoch_progress = (batch_count / train_dataloader_size) * 100
                overall_progress = (self.global_steps / self.total_training_steps) * 100
                samples_in_this_batch = len(batch_dict['input_ids'])
                epoch_samples_processed += samples_in_this_batch
                total_samples_processed += samples_in_this_batch
                
                # 时间预估
                if self.global_steps > 1:
                    avg_time_per_step = (time.time() - epoch_start_time + timing_raw.get('step', 0)) / batch_count if batch_count > 0 else 54.22
                    remaining_steps = self.total_training_steps - self.global_steps
                    eta_minutes = (remaining_steps * avg_time_per_step) / 60
                else:
                    eta_minutes = ((self.total_training_steps - self.global_steps) * 54.22) / 60
                
                print(f"⚡ Step {self.global_steps}/{self.total_training_steps}")
                print(f"📍 Epoch {epoch + 1}: Batch {batch_count}/{train_dataloader_size} ({epoch_progress:.1f}%)")
                print(f"📊 Samples: {samples_in_this_batch} this batch, {epoch_samples_processed}/{train_dataloader_size * actual_batch_size} this epoch, {total_samples_processed} total")
                print(f"⏰ ETA: {eta_minutes:.1f} minutes ({eta_minutes/60:.1f} hours)")
                
                metrics = {}

                # 数据加载阶段
                data_load_start = time.time()
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                data_load_time = time.time() - data_load_start
                logger.info(f"   📦 Batch loaded: {len(new_batch)} samples in {data_load_time:.2f}s")
                
                # 内存使用情况
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    logger.debug(f"   💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # 数据预处理阶段
                preprocess_start = time.time()
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                preprocess_time = time.time() - preprocess_start
                print(f"   🔧 Data preprocessing completed in {preprocess_time:.2f}s")

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # === 生成阶段 ===
                    gen_start_time = time.time()
                    print(f"   🎲 Starting generation phase...")
                    print(f"      🔄 Generating {self.config.actor_rollout_ref.rollout.n} responses per prompt (total: {len(gen_batch) * self.config.actor_rollout_ref.rollout.n} responses)...")
                    
                    with _timer("gen", timing_raw):
                        try:
                            breakpoint()
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                            gen_time = time.time() - gen_start_time
                            print(f"   ✅ Generation completed in {gen_time:.2f}s ({gen_time/len(gen_batch):.2f}s per prompt)")
                        except Exception as e:
                            print(f"   ❌ Generation failed: {e}")
                            raise

                    # REMAX baseline (如果启用)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        remax_start_time = time.time()
                        logger.info(f"   🔄 Computing REMAX baseline...")
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                        remax_time = time.time() - remax_start_time
                        print(f"   ✅ REMAX baseline computed in {remax_time:.2f}s")

                    # 批次数据合并
                    merge_start_time = time.time()
                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    merge_time = time.time() - merge_start_time
                    print(f"   🔗 Batch merging completed in {merge_time:.2f}s, final batch size: {len(new_batch)}")

                    # === 奖励计算阶段 ===
                    reward_start_time = time.time()
                    print(f"   🏆 Starting reward computation phase...")
                    print(f"      📊 Computing rewards for {len(new_batch)} samples...")
                    
                    with _timer("reward", timing_raw):
                        # Reward Model 评分
                        if self.use_rm:
                            rm_start_time = time.time()
                            print(f"      🤖 Computing reward model scores...")
                            try:
                                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(reward_tensor)
                                rm_time = time.time() - rm_start_time
                                print(f"      ✅ Reward model scores computed in {rm_time:.2f}s ({rm_time/len(new_batch)*1000:.1f}ms per sample)")
                            except Exception as e:
                                print(f"      ❌ Reward model computation failed: {e}")
                                raise

                        # Rule-based 奖励计算
                        rule_reward_start_time = time.time()
                        logger.info(f"      📊 Computing rule-based rewards (Wizard Reward)...")
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                            rule_reward_time = time.time() - rule_reward_start_time
                            logger.info(f"      ✅ Rule-based rewards computed in {rule_reward_time:.2f}s ({rule_reward_time/len(new_batch)*1000:.1f}ms per sample)")
                        except Exception as e:
                            logger.error(f"      ❌ Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        sample_scores = reward_tensor.sum(-1).cpu().tolist()
                        positive_scores = sum(1 for score in sample_scores if score > 0)
                        zero_scores = sum(1 for score in sample_scores if score == 0)
                        negative_scores = sum(1 for score in sample_scores if score < 0)
                        print(f"      📈 Sample scores: {len(sample_scores)} total")
                        print(f"         ✅ Positive: {positive_scores} ({positive_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         ⚪ Zero: {zero_scores} ({zero_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         ❌ Negative: {negative_scores} ({negative_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         📊 Average: {np.mean(sample_scores):.3f}")

                        # 思维词汇信息处理
                        if "thinking_tokens_info" in reward_result:
                            thinking_start_time = time.time()
                            logger.debug(f"      🧠 Processing thinking tokens info...")
                            thinking_tokens_infos_dict = reward_result["thinking_tokens_info"]
                            for key_info in list(thinking_tokens_infos_dict.keys()):
                                lst = thinking_tokens_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        thinking_tokens_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        thinking_tokens_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in thinking_tokens_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)
                            thinking_time = time.time() - thinking_start_time
                            logger.debug(f"      ✅ Thinking tokens processed in {thinking_time:.2f}s")

                        # 重复信息处理
                        if "repetition_info" in reward_result:
                            repetition_start_time = time.time()
                            logger.debug(f"      🔁 Processing repetition info...")
                            repetition_infos_dict = reward_result["repetition_info"]
                            for key_info in list(repetition_infos_dict.keys()):
                                lst = repetition_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        repetition_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        repetition_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in repetition_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)
                            repetition_time = time.time() - repetition_start_time
                            logger.debug(f"      ✅ Repetition info processed in {repetition_time:.2f}s")

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # KL惩罚
                        if self.config.algorithm.use_kl_in_reward:
                            kl_start_time = time.time()
                            logger.debug(f"      🎯 Applying KL penalty...")
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                            kl_time = time.time() - kl_start_time
                            logger.debug(f"      ✅ KL penalty applied in {kl_time:.2f}s")
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    reward_total_time = time.time() - reward_start_time
                    logger.info(f"   ✅ Reward computation completed in {reward_total_time:.2f}s")

                    # === 拒绝采样阶段 ===
                    sampling_start_time = time.time()
                    logger.info(f"   🎯 Starting rejection sampling...")
                    
                    # Group rewards by uid
                    uids = new_batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)
                    valid_mask = torch.ones(len(uids), dtype=torch.bool)
                    solve_none = 0
                    solve_all = 0
                    for uid in unique_uids:
                        uid_mask = uids == uid
                        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                        # Check if all rewards are 0 or all are 1 for this uid
                        if (uid_rewards == 0).all():
                            valid_mask[uid_mask] = False
                            solve_none += 1
                        elif (uid_rewards == 1).all():
                            valid_mask[uid_mask] = False
                            solve_all += 1

                    # Log to metrics
                    metrics['batch/solve_none'] = solve_none
                    metrics['batch/solve_all'] = solve_all
                    metrics['batch/valid'] = len(unique_uids) - solve_all - solve_none
                    valid_prompts = len(unique_uids) - solve_all - solve_none
                    logger.info(f"   📊 Rejection sampling results:")
                    logger.info(f"      🎯 Valid prompts: {valid_prompts}/{len(unique_uids)} ({valid_prompts/len(unique_uids)*100:.1f}%)")
                    logger.info(f"      ❌ Solve none: {solve_none} ({solve_none/len(unique_uids)*100:.1f}%)")
                    logger.info(f"      ✅ Solve all: {solve_all} ({solve_all/len(unique_uids)*100:.1f}%)")

                    if self.config.trainer.rejection_sample:
                        # If no valid samples remain, skip this batch and get a new one
                        if not valid_mask.any():
                            logger.warning(f"   ⚠️  No valid samples remaining, skipping batch...")
                            continue
                        # Filter batch to keep only valid samples
                        batch = new_batch[valid_mask]
                        logger.info(f"   ✅ Filtered to {len(batch)} valid samples for training")

                    # 长度过滤
                    length_filter_start = time.time()
                    max_response_length = batch.batch['responses'].shape[-1]
                    response_mask = batch.batch['attention_mask'][:, -max_response_length:]
                    response_length = response_mask.sum(-1).float()
                    response_clip_mask = ~torch.ge(response_length, max_response_length)
                    metrics['batch/clip_overlong'] = len(batch) - response_clip_mask.sum()
                    within_limit = response_clip_mask.sum().item()
                    logger.debug(f"   📏 Response length check: {within_limit}/{len(batch)} samples within length limit")
                    
                    if self.config.trainer.enable_overlong_filter:
                        batch = batch[response_clip_mask]
                        logger.debug(f"   ✂️  Filtered overlong responses, remaining: {len(batch)} samples")

                    # 批次大小调整
                    def get_sorted_indices(lst):
                        return [index for index, _ in sorted(enumerate(lst), key=lambda x: x[1])]
                    sorted_indices = torch.tensor(get_sorted_indices(batch.non_tensor_batch['index']))
                    batch.reorder(sorted_indices)

                    # Round down to the nearest multiple of world size
                    num_trainer_replicas = self.actor_rollout_wg.world_size
                    if batch.batch['input_ids'].shape[0] < num_trainer_replicas and num_trainer_replicas/batch.batch['input_ids'].shape[0] <= 2:
                        batch = batch.repeat(repeat_times=math.ceil(num_trainer_replicas/batch.batch['input_ids'].shape[0]), interleave=False)

                    max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                    if not max_batch_size:
                        logger.warning(f"   ⚠️  Batch size too small for {num_trainer_replicas} replicas, skipping...")
                        continue
                    batch = batch[:max_batch_size]
                    final_batch_size = len(batch)
                    logger.info(f"   ✅ Final training batch size: {final_batch_size} samples")
                    sampling_time = time.time() - sampling_start_time
                    logger.info(f"   ✅ Rejection sampling completed in {sampling_time:.2f}s")

                    # === 模型更新阶段 ===
                    update_start_time = time.time()
                    logger.info(f"   🔄 Starting model update phase with {final_batch_size} samples...")

                    # Response mask计算
                    mask_start_time = time.time()
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    mask_time = time.time() - mask_start_time
                    logger.debug(f"      🎭 Response mask computed in {mask_time:.2f}s")

                    # Batch balancing
                    if self.config.trainer.balance_batch:
                        balance_start_time = time.time()
                        logger.debug(f"      ⚖️  Balancing batch across DP ranks...")
                        self._balance_batch(batch, metrics=metrics)
                        balance_time = time.time() - balance_start_time
                        logger.debug(f"      ✅ Batch balanced in {balance_time:.2f}s")

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Old log probabilities计算
                    old_logprob_start_time = time.time()
                    logger.debug(f"      📊 Computing old log probabilities...")
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                    old_logprob_time = time.time() - old_logprob_start_time
                    logger.debug(f"      ✅ Old log probabilities computed in {old_logprob_time:.2f}s")

                    # Reference policy
                    if self.use_reference_policy:
                        ref_start_time = time.time()
                        logger.debug(f"      🔗 Computing reference log probabilities...")
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        ref_time = time.time() - ref_start_time
                        logger.debug(f"      ✅ Reference log probabilities computed in {ref_time:.2f}s")

                    # Critic values
                    if self.use_critic:
                        values_start_time = time.time()
                        logger.debug(f"      💰 Computing values...")
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        values_time = time.time() - values_start_time
                        logger.debug(f"      ✅ Values computed in {values_time:.2f}s")

                    # Advantages计算
                    adv_start_time = time.time()
                    logger.debug(f"      🎯 Computing advantages...")
                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )
                        if "inverse_pair" in batch.meta_info:
                            metrics["batch/inverse_pair"] = batch.meta_info["inverse_pair"] / metrics["batch/valid"]
                    adv_time = time.time() - adv_start_time
                    logger.debug(f"      ✅ Advantages computed in {adv_time:.2f}s")

                    # === 关键的训练更新部分 ===
                    # Critic更新
                    if self.use_critic:
                        critic_update_start = time.time()
                        logger.info(f"      🎓 Updating critic network...")
                        logger.info(f"         📊 Processing {final_batch_size} samples for critic training")
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                        critic_update_time = time.time() - critic_update_start
                        logger.info(f"      ✅ Critic updated in {critic_update_time:.2f}s")
                        
                        # 记录critic的关键指标
                        if 'critic/loss' in critic_output_metrics:
                            logger.info(f"         📉 Critic loss: {critic_output_metrics['critic/loss']:.4f}")

                    # Actor更新 - 这是最关键的训练更新部分
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        actor_update_start = time.time()
                        logger.info(f"      🎭 Updating actor network (main training update)...")
                        logger.info(f"         📊 Processing {final_batch_size} samples for actor training")
                        logger.info(f"         🔄 PPO epochs: {self.config.actor_rollout_ref.actor.ppo_epochs}")
                        logger.info(f"         📦 Mini-batch size: {self.config.actor_rollout_ref.actor.ppo_mini_batch_size}")
                        
                        # 计算会有多少个mini-batches
                        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                        num_mini_batches = math.ceil(final_batch_size / ppo_mini_batch_size)
                        total_ppo_updates = num_mini_batches * self.config.actor_rollout_ref.actor.ppo_epochs
                        logger.info(f"         🔢 Will perform {total_ppo_updates} mini-batch updates ({num_mini_batches} mini-batches × {self.config.actor_rollout_ref.actor.ppo_epochs} epochs)")
                        
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        actor_update_time = time.time() - actor_update_start
                        
                        # 详细的actor更新信息
                        logger.info(f"      ✅ Actor updated in {actor_update_time:.2f}s ({actor_update_time/total_ppo_updates*1000:.1f}ms per update)")
                        
                        # 记录actor的关键指标
                        if 'actor/loss' in actor_output_metrics:
                            logger.info(f"         📉 Actor loss: {actor_output_metrics['actor/loss']:.4f}")
                        if 'actor/lr' in actor_output_metrics:
                            logger.info(f"         📈 Learning rate: {actor_output_metrics['actor/lr']:.6f}")
                        if 'actor/entropy' in actor_output_metrics:
                            logger.info(f"         🎲 Entropy: {actor_output_metrics['actor/entropy']:.4f}")
                            
                    else:
                        remaining_warmup = self.config.trainer.critic_warmup - self.global_steps
                        logger.info(f"      ⏳ Critic warmup phase: {self.global_steps}/{self.config.trainer.critic_warmup} (remaining: {remaining_warmup} steps)")

                    update_total_time = time.time() - update_start_time
                    logger.info(f"   ✅ Model update completed in {update_total_time:.2f}s")

                    # 检查点保存
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        save_start_time = time.time()
                        logger.info(f"   💾 Saving checkpoint at step {self.global_steps}...")
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        save_time = time.time() - save_start_time
                        logger.info(f"   ✅ Checkpoint saved in {save_time:.2f}s")

                    # 验证
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        val_start_time = time.time()
                        logger.info(f"   🧪 Running validation...")
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        val_time = time.time() - val_start_time
                        logger.info(f"   ✅ Validation completed in {val_time:.2f}s")

                # 指标收集
                metrics_start_time = time.time()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                metrics_time = time.time() - metrics_start_time
                logger.debug(f"   📊 Metrics computed in {metrics_time:.2f}s")

                # 步骤总结
                step_total_time = time.time() - step_start_time
                samples_per_second = samples_in_this_batch / step_total_time
                estimated_total_time = step_total_time * (self.total_training_steps - self.global_steps + 1)
                
                logger.info(f"✅ Step {self.global_steps} completed in {step_total_time:.2f}s")
                logger.info(f"📈 Overall progress: {self.global_steps}/{self.total_training_steps} ({overall_progress:.1f}%)")
                logger.info(f"⚡ Processing speed: {samples_per_second:.2f} samples/second")
                logger.info(f"⏱️  Estimated remaining time: {estimated_total_time/3600:.1f} hours")

                # WandB日志记录
                tracking_logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    logger.info("🎉 Training completed successfully!")
                    print("📊 Final validation metrics:")
                    pprint(last_val_metrics)
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                
            epoch_total_time = time.time() - epoch_start_time
            epoch_samples_per_second = epoch_samples_processed / epoch_total_time
            logger.info(f"📚 ==================== Epoch {epoch + 1} Summary ====================")
            logger.info(f"⏱️  Epoch completed in {epoch_total_time/60:.1f} minutes")
            logger.info(f"📊 Processed {epoch_samples_processed} samples")
            logger.info(f"⚡ Average speed: {epoch_samples_per_second:.2f} samples/second")
            logger.info(f"📈 Completed {batch_count}/{train_dataloader_size} batches")
            logger.info(f"==================================================================")
