
<div align="center">

# ✨ Archer

<div>
🏹️  Reinforcement Learning for Enhanced Reasoning in LLMs  🎯
</div>

</div>
<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/wizard-III/ArcherCodeR)
[![Model](https://img.shields.io/badge/Model-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/Fate-Zero/Archer-Code-1.5B)
[![Data](https://img.shields.io/badge/Data-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/datasets/Fate-Zero/Archer-Code-1.5B)
[![Wandb](https://img.shields.io/badge/Wandb-000000?style=for-the-badge&logo=Wandb&logoColor=000&labelColor)](https://wandb.ai/wangjkpkucs-peking-university/ArcherCodeR?nw=nwuserwangjkpkucs)
[![知乎](https://img.shields.io/badge/知乎-0084FF?style=for-the-badge&logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/1918765619614057424)

</div>

## Overview

The Archer series focuses on research into RL algorithms and training for medium and small-scale models, aiming to deepen the community's understanding of the fundamental principles of reinforcement learning (RL) on large language models (LLMs). All released content will be comprehensively open-sourced to advance community research development.

<div align="center">
<img src="assets/combined_math_code_benchmarks.png" width="100%"/>

<sub>Archer significantly improves the reasoning performance upon DAPO and outperforms previous 1.5B-level SOTA reasoning models.</sub>
</div>

**Archer** is an open-source initiative enhancing reasoning in large language models through scalable, rule-governed reinforcement learning. We provide full-stack reproducibility including:

- Training code and pipelines
- Curated datasets
- Trained models
- Complete training logs

**Current Models**:
- **[Archer-Code-1.5B](https://huggingface.co/Fate-Zero/Archer-Code-1.5B)** - SOTA among similarly-sized models.

## Evaluation
We conduct evaluation on both mathematical and coding benchmarks. Due to the high variance of the outputs from reasoning models, we report avg@K (pass@1 performance averaged over K outputs) and pass@K for each benchmark. The detailed results are shown in the table below.


<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">AIME24</th>
      <th colspan="2">AIME25</th>
      <th colspan="2">AMC23</th>
      <th colspan="2">MATH-500</th>
      <th colspan="2">Minerva</th>
      <th colspan="2">Olympiad</th>
      <th rowspan="2">Avg.</th>
    </tr>
    <tr>
      <th>avg@64</th>
      <th>pass@64</th>
      <th>avg@64</th>
      <th>pass@64</th>
      <th>avg@64</th>
      <th>pass@64</th>
      <th>avg@4</th>
      <th>pass@4</th>
      <th>avg@8</th>
      <th>pass@8</th>
      <th>avg@4</th>
      <th>pass@4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DeepSeek-R1-1.5B</td>
      <td>30.6</td><td>80.0</td>
      <td>23.5</td><td>63.3</td>
      <td>70.7</td><td>100.0</td>
      <td>83.6</td><td>92.4</td>
      <td>27.6</td><td>48.2</td>
      <td>44.6</td><td>59.4</td>
      <td>46.8</td>
    </tr>
    <tr>
      <td>DAPO</td>
      <td>42.1</td><td>80.0</td>
      <td>28.6</td><td>56.7</td>
      <td>80.3</td><td>97.5</td>
      <td>87.6</td><td>94.6</td>
      <td>29.2</td><td>46.3</td>
      <td>53.2</td><td>65.8</td>
      <td>53.5</td>
    </tr>
    <tr>
      <td>DeepScaleR-1.5B</td>
      <td>42.0</td><td><strong>83.3</strong></td>
      <td>29.0</td><td>63.3</td>
      <td>81.3</td><td>100.0</td>
      <td>87.7</td><td>93.6</td>
      <td>30.3</td><td>51.1</td>
      <td>50.7</td><td>61.0</td>
      <td>53.5</td>
    </tr>
    <tr>
      <td>FastCuRL-1.5B-V3</td>
      <td>48.1</td><td>80.0</td>
      <td>32.7</td><td>60.0</td>
      <td><strong>86.4</strong></td><td>95.0</td>
      <td>89.8</td><td>94.0</td>
      <td>33.6</td><td>50.0</td>
      <td>55.3</td><td>64.3</td>
      <td>57.7</td>
    </tr>
    <tr>
      <td>Nemotron-1.5B</td>
      <td>48.0</td><td>76.7</td>
      <td>33.1</td><td>60.0</td>
      <td>86.1</td><td>97.5</td>
      <td>90.6</td><td>93.6</td>
      <td>35.3</td><td>47.8</td>
      <td>59.2</td><td>66.8</td>
      <td>58.7</td>
    </tr>
    <tr>
      <td><strong>Archer-Math-1.5B</strong></td>
      <td><strong>48.7</strong></td><td><strong>83.3</strong></td>
      <td><strong>33.8</strong></td><td><strong>70.0</strong></td>
      <td>86.0</td><td><strong>97.5</strong></td>
      <td><strong>90.8</strong></td><td><strong>94.4</strong></td>
      <td><strong>35.7</strong></td><td><strong>51.1</strong></td>
      <td><strong>59.3</strong></td><td><strong>67.1</strong></td>
      <td><strong>59.1</strong></td>
    </tr>
  </tbody>
</table>


<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">LCB v5 (2024.08.01–2025.02.01)</th>
      <th colspan="2">LCB v6 (2025.02.01–2025.05.01)</th>
      <th rowspan="2">Avg.</th>
    </tr>
    <tr>
      <th>avg@8</th>
      <th>pass@8</th>
      <th>avg@16</th>
      <th>pass@16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DeepSeek-R1-1.5B</td>
      <td>16.7</td>
      <td>29.0</td>
      <td>17.2</td>
      <td>34.4</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>DAPO</td>
      <td>26.0</td>
      <td>40.5</td>
      <td>27.6</td>
      <td>43.5</td>
      <td>26.8</td>
    </tr>
    <tr>
      <td>DeepCoder-1.5B</td>
      <td>23.3</td>
      <td>39.1</td>
      <td>22.6</td>
      <td>42.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <td>Nemotron-1.5B</td>
      <td>26.1</td>
      <td>35.5</td>
      <td>29.5</td>
      <td>42.8</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td><strong>Archer-Code-1.5B</strong></td>
      <td><strong>29.4</strong></td>
      <td><strong>43.7</strong></td>
      <td><strong>30.2</strong></td>
      <td><strong>45.8</strong></td>
      <td><strong>29.8</strong></td>
    </tr>
  </tbody>
</table>

<!-- Note:
1. Evaluation variance for the same model is typically within ±0.5 across multiple runs.
2. DeepCoder consistently scored around 23 in our tests - lower than its reported performance.
3. NVIDIA's Nemotron-Research-Reasoning-Qwen-1.5B slightly outperformed its reported score, potentially due to different parameter settings in their original evaluation. -->

## Getting Started

### Installation

```bash
# Installing Python 3.10 Environment.
conda create -n archer python=3.10 -y
conda activate archer

# Installing dependencies.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cd ArcherCodeR
pip install -e .
```

### Data Preparation

Download the training and test data from Hugging Face.

```bash
python tools/download_datasets.py
```

#### Initialize Ray Cluster

We have provided a one-click script to initialize Ray environments on any number of machines. Run the following command on the head node:

```bash
bash ./tools/start_ray.sh
```

Note: 
- Please replace your_wandb_api_key in export WANDB_API_KEY=your_wandb_api_key with your actual key.
- Hostfile locations vary across operating systems (e.g., on my machine, it's located at /etc/mpi/hostfile). Locate the file on your server and modify its content accordingly.

### Training

We have currently only provided the script and data to reproduce the results of the “ArcherCodeR-1.5B-DAPO”.

```bash
bash ./scripts/train/run_archer_qwen2.5_1.5b_code.sh
```

### Evaluation

#### Step 1: Convert model format

Run the following command to convert the model to Hugging Face format:

```bash
bash ./tools/model_merge.sh
```

#### Step 2: Run evaluation

Execute the script below to evaluate model performance on the LiveCodeBench v5 benchmark:

```bash
bash ./scripts/eval/run_eval.sh
```

Note: Please update the path parameters in the scripts above as needed.

## Technical Report

[Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR](https://arxiv.org/abs/2507.15778)

## Acknowledgements

- We build our model upon [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- Training was carried out with a modified version of [verl](https://github.com/volcengine/verl).

## Citation

Please cite the following:

```bibtex
@misc{wang2025stabilizingknowledgepromotingreasoning,
      title={Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR}, 
      author={Jiakang Wang and Runze Liu and Fuzheng Zhang and Xiu Li and Guorui Zhou},
      year={2025},
      eprint={2507.15778},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.15778}, 
}
```

