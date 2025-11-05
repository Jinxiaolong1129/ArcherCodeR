#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:17:50
@Author  :   wangjiakang
@File    :   dapo_ray_trainer.py
'''


import os
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
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

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
        self.gen_steps = 0

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

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()
            
        print("🎯 Starting main training loop setup of RayDAPOTrainer...")
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress DAPO")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
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
                
                # 数据预处理阶段 - 与官方实现保持一致
                preprocess_start = time.time()
                # pop those keys for generation
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
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                preprocess_time = time.time() - preprocess_start
                print(f"   🔧 Data preprocessing completed in {preprocess_time:.2f}s")

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # === 生成阶段 ===
                    gen_start_time = time.time()
                    print(f"   🎲 Starting generation phase...")
                    print(f"      🔄 Generating {self.config.actor_rollout_ref.rollout.n} responses per prompt (total: {len(gen_batch_output)} responses)...")
                    
                    with marked_timer("gen", timing_raw, "red"):
                        try:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                            gen_time = time.time() - gen_start_time
                            print(f"   ✅ Generation completed in {gen_time:.2f}s ({gen_time/len(gen_batch_output):.2f}s per response)")
                        except Exception as e:
                            print(f"   ❌ Generation failed: {e}")
                            raise

                    # REMAX baseline (如果启用) - 与官方实现保持一致
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        remax_start_time = time.time()
                        logger.info(f"   🔄 Computing REMAX baseline...")
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
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

                    # KL相关指标计算 (如果在奖励中使用KL)
                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    # === 奖励计算阶段 ===
                    reward_start_time = time.time()
                    print(f"   🏆 Starting reward computation phase...")
                    print(f"      📊 Computing rewards for {len(new_batch)} samples...")
                    
                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
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

                        # we combine with rule-based rm (保留自定义奖励函数逻辑)
                        rule_reward_start_time = time.time()
                        logger.info(f"      📊 Computing rule-based rewards (Custom Reward)...")
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            # 使用官方的compute_reward接口，但保留自定义奖励函数
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                            rule_reward_time = time.time() - rule_reward_start_time
                            logger.info(f"      ✅ Rule-based rewards computed in {rule_reward_time:.2f}s ({rule_reward_time/len(new_batch)*1000:.1f}ms per sample)")
                        except Exception as e:
                            logger.error(f"      ❌ Error in reward_fn: {e}")
                            # 回退到直接调用自定义奖励函数
                            try:
                                reward_result = self.reward_fn(new_batch, return_dict=True)
                                reward_tensor = reward_result["reward_tensor"]
                                reward_extra_infos_dict = reward_result["reward_extra_info"]
                            except:
                                reward_tensor = self.reward_fn(new_batch)
                                reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            kl_start_time = time.time()
                            logger.debug(f"      🎯 Applying KL penalty...")
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                            kl_time = time.time() - kl_start_time
                            logger.debug(f"      ✅ KL penalty applied in {kl_time:.2f}s")
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        # 显示奖励统计信息 (保留自定义日志)
                        sample_scores = reward_tensor.sum(-1).cpu().tolist()
                        positive_scores = sum(1 for score in sample_scores if score > 0)
                        zero_scores = sum(1 for score in sample_scores if score == 0)
                        negative_scores = sum(1 for score in sample_scores if score < 0)
                        print(f"      📈 Sample scores: {len(sample_scores)} total")
                        print(f"         ✅ Positive: {positive_scores} ({positive_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         ⚪ Zero: {zero_scores} ({zero_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         ❌ Negative: {negative_scores} ({negative_scores/len(sample_scores)*100:.1f}%)")
                        print(f"         📊 Average: {np.mean(sample_scores):.3f}")

                    reward_total_time = time.time() - reward_start_time
                    logger.info(f"   ✅ Reward computation completed in {reward_total_time:.2f}s")

                    # === 官方DAPO动态采样算法 ===
                    sampling_start_time = time.time()
                    logger.info(f"   🎯 Starting DAPO dynamic sampling...")
                    
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                        logger.info(f"   📊 Dynamic sampling disabled, using all {len(batch)} samples")
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            logger.info(f"   📊 Dynamic sampling: {num_prompt_in_batch} < {prompt_bsz} prompts")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                logger.info(f"   🔄 Continue generating (batch {num_gen_batches})...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                            logger.info(f"   ✅ Dynamic sampling completed: {len(batch)} samples selected")

                    sampling_time = time.time() - sampling_start_time
                    logger.info(f"   ✅ DAPO dynamic sampling completed in {sampling_time:.2f}s")

                    # === 更新阶段 ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    update_start_time = time.time()
                    logger.info(f"   🔄 Starting model update phase with {len(batch)} samples...")

                    if self.config.trainer.balance_batch:
                        balance_start_time = time.time()
                        logger.debug(f"      ⚖️  Balancing batch across DP ranks...")
                        self._balance_batch(batch, metrics=metrics)
                        balance_time = time.time() - balance_start_time
                        logger.debug(f"      ✅ Batch balanced in {balance_time:.2f}s")

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        values_start_time = time.time()
                        logger.debug(f"      💰 Computing values...")
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        values_time = time.time() - values_start_time
                        logger.debug(f"      ✅ Values computed in {values_time:.2f}s")

                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)

                    # Advantages计算
                    adv_start_time = time.time()
                    logger.debug(f"      🎯 Computing advantages...")
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    adv_time = time.time() - adv_start_time
                    logger.debug(f"      ✅ Advantages computed in {adv_time:.2f}s")

                    # update critic
                    if self.use_critic:
                        critic_update_start = time.time()
                        logger.info(f"      🎓 Updating critic network...")
                        logger.info(f"         📊 Processing {len(batch)} samples for critic training")
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                        critic_update_time = time.time() - critic_update_start
                        logger.info(f"      ✅ Critic updated in {critic_update_time:.2f}s")
                        
                        # 记录critic的关键指标
                        if 'critic/loss' in critic_output_metrics:
                            logger.info(f"         📉 Critic loss: {critic_output_metrics['critic/loss']:.4f}")

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        actor_update_start = time.time()
                        logger.info(f"      🎭 Updating actor network (main training update)...")
                        logger.info(f"         📊 Processing {len(batch)} samples for actor training")
                        logger.info(f"         🔄 PPO epochs: {self.config.actor_rollout_ref.actor.ppo_epochs}")
                        logger.info(f"         📦 Mini-batch size: {self.config.actor_rollout_ref.actor.ppo_mini_batch_size}")
                        
                        # 计算会有多少个mini-batches
                        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                        num_mini_batches = math.ceil(len(batch) / ppo_mini_batch_size)
                        total_ppo_updates = num_mini_batches * self.config.actor_rollout_ref.actor.ppo_epochs
                        logger.info(f"         🔢 Will perform {total_ppo_updates} mini-batch updates ({num_mini_batches} mini-batches × {self.config.actor_rollout_ref.actor.ppo_epochs} epochs)")
                        
                        with marked_timer("update_actor", timing_raw, "red"):
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

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    update_total_time = time.time() - update_start_time
                    logger.info(f"   ✅ Model update completed in {update_total_time:.2f}s")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    val_start_time = time.time()
                    logger.info(f"   🧪 Running validation...")
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    val_time = time.time() - val_start_time
                    logger.info(f"   ✅ Validation completed in {val_time:.2f}s")

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    save_start_time = time.time()
                    logger.info(f"   💾 Saving checkpoint at step {self.global_steps}...")
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()
                    save_time = time.time() - save_start_time
                    logger.info(f"   ✅ Checkpoint saved in {save_time:.2f}s")

                # collect metrics
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
                self.gen_steps += 1
                
            epoch_total_time = time.time() - epoch_start_time
            epoch_samples_per_second = epoch_samples_processed / epoch_total_time
            logger.info(f"📚 ==================== Epoch {epoch + 1} Summary ====================")
            logger.info(f"⏱️  Epoch completed in {epoch_total_time/60:.1f} minutes")
            logger.info(f"📊 Processed {epoch_samples_processed} samples")
            logger.info(f"⚡ Average speed: {epoch_samples_per_second:.2f} samples/second")
            logger.info(f"📈 Completed {batch_count}/{train_dataloader_size} batches")
            logger.info(f"==================================================================")
        
        # check if last step checkpint exists (官方实现的最后检查点保存逻辑)
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            tracking_logger.log(data=metrics, step=self.global_steps)
