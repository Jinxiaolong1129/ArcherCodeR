# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
from functools import partial

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine=0, for_validation=False, **reward_kwargs):
    """
    Load and instantiate a reward manager based on the provided configuration.

    Args:
        config: Configuration object containing reward model settings.
        tokenizer: Tokenizer instance for processing text.
        num_examine (int, optional): Number of batches to examine for debugging. Defaults to 0.
        for_validation (bool, optional): Whether this is for validation. Defaults to False.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Check if we should use rewards/general_reward.py (like DAPO)
    use_general_reward = config.reward_model.get("use_general_reward", False)
    
    # Special handling for Intuitor algorithm
    if hasattr(config, 'algorithm') and hasattr(config.algorithm, 'adv_estimator') and config.algorithm.adv_estimator == "intuitor":
        if for_validation:
            # For validation, use actual reward function (e.g., livecodebench) - import directly like DAPO
            print("🎯 Loading general_reward_fn for Intuitor validation (supports livecodebench)")
            from rewards.general_reward import general_reward_fn
            final_compute_score = general_reward_fn
        else:
            # For training, use dummy reward since Intuitor uses self-certainty
            print("🎯 Loading dummy reward function for Intuitor training (uses self-certainty)")
            def dummy_reward_fn(data_source: str, solution_str: str, ground_truth, extra_info=None, enable_llm=False, is_eval=False):
                return 0.0
            final_compute_score = dummy_reward_fn
    elif use_general_reward:
        # Use rewards/general_reward.py like DAPO does
        print("🎯 Loading general_reward_fn from rewards/ package (DAPO-style)")
        from rewards.general_reward import general_reward_fn
        final_compute_score = general_reward_fn
    else:
        # Try to get a custom reward function based on the configuration
        compute_score = get_custom_reward_fn(config)
        final_compute_score = compute_score

        if compute_score is None:
            sandbox_config = config.reward_model.get("sandbox_fusion")
            sandbox_url = sandbox_config.get("url") if sandbox_config else None
            if sandbox_url:
                sandbox_manager = multiprocessing.Manager()
                # Create a semaphore to control concurrent access to the sandbox
                _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
                final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
            else:
                final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
