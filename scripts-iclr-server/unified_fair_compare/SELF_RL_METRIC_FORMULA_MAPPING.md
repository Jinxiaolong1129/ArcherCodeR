# Self-RL Metrics Formula Mapping

This document records the final metric-to-formula mapping used in the current implementation for unified fair-comparison experiments.

## Scope

- Training entry: `verl.trainer.main_ppo`
- Metric computation: `verl/trainer/ppo/metric_utils.py`
- Advantage computation branches: `verl/trainer/ppo/ray_trainer.py` and `verl/trainer/ppo/core_algos.py`
- Focus methods: `intuitor`, `token_entropy`, `trajectory_entropy`, `prob_disparity`, `grpo`

---

## 0. Notation

- \(i\): response index (one sampled response)
- \(t\): token index inside one response
- \(M_{i,t} \in \{0,1\}\): training response mask (`training_response_mask`)
- \(L_i = \sum_t M_{i,t}\): valid response-token count for response \(i\)
- \(u(i)\): prompt group id (`uid`) for response \(i\)
- \(\mathcal{G}_u\): set of responses that share the same uid
- \(\epsilon = 10^{-8}\)

---

## 1. Core sequence-level operator

### 1.1 Per-sequence masked mean

\[
\text{SeqMean}_i(X) = \frac{\sum_t X_{i,t} M_{i,t}}{\sum_t M_{i,t} + \epsilon}
\]

### 1.2 Prompt-group std mean (aligned with training normalization)

For each prompt group \(\mathcal{G}_u\), define group std:

- If \(|\mathcal{G}_u|=1\): \(\sigma_u=1.0\)
- If \(|\mathcal{G}_u|>1\): \(\sigma_u=\text{Std}_{sample}(\{s_i\}_{i\in\mathcal{G}_u})\) (`ddof=1`)

Then:

\[
\text{GroupStdMean}(s) = \frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}} \sigma_u
\]

---

## 2. `monitor/*` metrics (cross-method primary monitoring)

These are sequence-level raw monitoring signals.

### 2.1 Self-certainty

\[
s_i^{self} = \text{SeqMean}_i(\texttt{self\_certaintys})
\]

- `monitor/seq_self_certainty_mean` = \(\text{Mean}_i(s_i^{self})\)
- `monitor/seq_self_certainty_std` = \(\text{Std}_i(s_i^{self})\)
- `monitor/prompt_group_self_certainty_std_mean` = \(\text{GroupStdMean}(s^{self})\)

### 2.2 Token entropy

\[
s_i^{tokent} = \text{SeqMean}_i(\texttt{entropys})
\]

- `monitor/seq_token_entropy_mean` = \(\text{Mean}_i(s_i^{tokent})\)
- `monitor/seq_token_entropy_std` = \(\text{Std}_i(s_i^{tokent})\)
- `monitor/prompt_group_token_entropy_std_mean` = \(\text{GroupStdMean}(s^{tokent})\)

### 2.3 Trajectory score (training-sign aligned)

\[
s_i^{traj} = \text{SeqMean}_i(\texttt{old\_log\_probs})
\]

- `monitor/seq_trajectory_score_mean` = \(\text{Mean}_i(s_i^{traj})\)
- `monitor/seq_trajectory_score_std` = \(\text{Std}_i(s_i^{traj})\)
- `monitor/prompt_group_trajectory_score_std_mean` = \(\text{GroupStdMean}(s^{traj})\)

### 2.4 Probability disparity

\[
s_i^{pdisp} = \text{SeqMean}_i(\texttt{prob\_disparitys})
\]

- `monitor/seq_prob_disparity_mean` = \(\text{Mean}_i(s_i^{pdisp})\)
- `monitor/seq_prob_disparity_std` = \(\text{Std}_i(s_i^{pdisp})\)
- `monitor/prompt_group_prob_disparity_std_mean` = \(\text{GroupStdMean}(s^{pdisp})\)

---

## 3. `selfrl/*` metrics (current-estimator training signal)

### 3.1 Raw sequence score \(r_i\) by `adv_estimator`

- `intuitor`, `intuitor_selective`:
  \[
  r_i = s_i^{self}
  \]

- `token_entropy`, `intuitor_entropy`:
  \[
  r_i = -s_i^{tokent}
  \]

- `trajectory_entropy`:
  \[
  r_i = s_i^{traj}
  \]

- `prob_disparity`:
  \[
  r_i = s_i^{pdisp}
  \]

Mapped metrics:

- `selfrl/seq_score_raw_mean` = \(\text{Mean}_i(r_i)\)
- `selfrl/seq_score_raw_std` = \(\text{Std}_i(r_i)\)
- `selfrl/prompt_group_std_mean` = \(\text{GroupStdMean}(r)\)

### 3.2 Normalized sequence score (from final advantages)

Let \(A_{i,t}\) be final token-level advantage used by PPO update:

\[
z_i = \text{SeqMean}_i(A)
\]

- `selfrl/seq_score_norm_mean` = \(\text{Mean}_i(z_i)\)
- `selfrl/seq_score_norm_std` = \(\text{Std}_i(z_i)\)

### 3.3 Token-level advantage magnitude

\[
\texttt{selfrl/adv\_token\_mean\_abs}
= \frac{\sum_{i,t} |A_{i,t}| M_{i,t}}{\sum_{i,t} M_{i,t}}
\]

If no valid training token exists, value is set to `0.0`.

---

## 4. `internal_metrics/*` (auxiliary sanity-check metrics)

These are token-level aggregates. They are useful for health checks and drift monitoring, but they are not the direct final training signal fed to loss.

- `internal_metrics/self_certainty/*`
- `internal_metrics/token_entropy/*`
- `internal_metrics/trajectory_entropy/*` (token-level on `-old_log_probs`)
- `internal_metrics/prob_disparity/*`

---

## 5. Practical interpretation

- For cross-method analysis, prioritize:
  - `monitor/seq_*`
  - `monitor/prompt_group_*_std_mean`
  - `selfrl/seq_score_raw_*`, `selfrl/seq_score_norm_*`, `selfrl/adv_token_mean_abs`
- Use `internal_metrics/*` as auxiliary sanity checks.

