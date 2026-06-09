# Phase 1 v2 analysis - anisotropy-corrected metrics

- Qwen N=500   mean_pairwise_cos(gold) = **0.4969**  (124,750 pairs)
- Gemma N=500  mean_pairwise_cos(gold) = **0.9746**  (124,750 pairs)

Methodological note: a raw MSE = 2(1−cos) comparison would credit Gemma for ~0.07 of free MSE just by residual-stream geometry. ABC subtracts that baseline so both models are scored on "how much beyond random gold-pair similarity" the AR captures.

## Per-subgroup metrics (lead: ABC)

### Qwen2.5-7B  (layer 20, d=3584) - per-subgroup metrics

`mean_pairwise_cos(gold) = 0.4969` (over all 124,750 pairs in the 500-input set)

| sub | n | ABC mean | ABC median | ABC stdev | cos mean | raw MSE mean |
|---|---|---|---|---|---|---|
| A | 100 | +0.371 | +0.376 | 0.034 | 0.868 | 0.264 |
| B | 100 | +0.300 | +0.306 | 0.041 | 0.797 | 0.407 |
| C | 100 | +0.306 | +0.316 | 0.038 | 0.802 | 0.395 |
| D | 100 | +0.346 | +0.348 | 0.030 | 0.843 | 0.313 |
| E1 | 30 | +0.247 | +0.258 | 0.063 | 0.744 | 0.511 |
| E2 | 70 | +0.406 | +0.409 | 0.024 | 0.903 | 0.194 |

### Gemma3-12B  (layer 32, d=3840) - per-subgroup metrics

`mean_pairwise_cos(gold) = 0.9746` (over all 124,750 pairs in the 500-input set)

| sub | n | ABC mean | ABC median | ABC stdev | cos mean | raw MSE mean |
|---|---|---|---|---|---|---|
| A | 100 | +0.022 | +0.022 | 0.001 | 0.996 | 0.007 |
| B | 100 | +0.013 | +0.014 | 0.004 | 0.988 | 0.024 |
| C | 100 | +0.022 | +0.022 | 0.001 | 0.996 | 0.007 |
| D | 100 | +0.017 | +0.018 | 0.004 | 0.992 | 0.017 |
| E1 | 30 | +0.017 | +0.018 | 0.005 | 0.992 | 0.017 |
| E2 | 70 | +0.021 | +0.021 | 0.001 | 0.996 | 0.009 |

## Pairwise tests on ABC

### Qwen2.5-7B  (layer 20, d=3584) - pairwise Welch's t-test on ABC
Bonferroni α = 0.05/15 = 0.0033.  Cohen's d shown for the requested A↔E1, A↔E2, E1↔E2 contrasts.

| group1 | group2 | mean1 | mean2 | Δ (g2−g1) | t | df | p | sig |
|---|---|---|---|---|---|---|---|---|
| A | B | +0.371 | +0.300 | -0.072 | +13.33 | 191.9 | 4.11e-29 | **yes** |
| A | C | +0.371 | +0.306 | -0.066 | +12.81 | 196.1 | 1.07e-27 | **yes** |
| A | D | +0.371 | +0.346 | -0.025 | +5.47 | 193.6 | 1.39e-07 | **yes** |
| A | E1 | +0.371 | +0.247 | -0.124 | +10.25 | 34.3 | 5.66e-12 | **yes** |
| A | E2 | +0.371 | +0.406 | +0.035 | -7.80 | 168.0 | 6.22e-13 | **yes** |
| B | C | +0.300 | +0.306 | +0.006 | -1.04 | 196.7 | 2.97e-01 | no |
| B | D | +0.300 | +0.346 | +0.047 | -9.21 | 179.7 | 8.36e-17 | **yes** |
| B | E1 | +0.300 | +0.247 | -0.052 | +4.24 | 36.7 | 1.44e-04 | **yes** |
| B | E2 | +0.300 | +0.406 | +0.107 | -21.25 | 162.8 | 8.28e-49 | **yes** |
| C | D | +0.306 | +0.346 | +0.041 | -8.49 | 186.8 | 6.48e-15 | **yes** |
| C | E1 | +0.306 | +0.247 | -0.058 | +4.76 | 35.5 | 3.23e-05 | **yes** |
| C | E2 | +0.306 | +0.406 | +0.101 | -21.17 | 166.2 | 4.68e-49 | **yes** |
| D | E1 | +0.346 | +0.247 | -0.099 | +8.28 | 32.9 | 1.51e-09 | **yes** |
| D | E2 | +0.346 | +0.406 | +0.060 | -14.53 | 164.7 | 2.55e-31 | **yes** |
| E1 | E2 | +0.247 | +0.406 | +0.159 | -13.31 | 32.6 | 1.02e-14 | **yes** |

**Cohen's d (ABC) for Qwen2.5-7B  (layer 20, d=3584) requested contrasts:**
- A vs E1:  d = +2.43  (mean_A=+0.371, mean_E1=+0.247)
- A vs E2:  d = -1.18  (mean_A=+0.371, mean_E2=+0.406)
- E1 vs E2:  d = -3.31  (mean_E1=+0.247, mean_E2=+0.406)

### Gemma3-12B  (layer 32, d=3840) - pairwise Welch's t-test on ABC
Bonferroni α = 0.05/15 = 0.0033.  Cohen's d shown for the requested A↔E1, A↔E2, E1↔E2 contrasts.

| group1 | group2 | mean1 | mean2 | Δ (g2−g1) | t | df | p | sig |
|---|---|---|---|---|---|---|---|---|
| A | B | +0.022 | +0.013 | -0.008 | +21.36 | 104.5 | 6.63e-40 | **yes** |
| A | C | +0.022 | +0.022 | -0.000 | +0.55 | 197.4 | 5.80e-01 | no |
| A | D | +0.022 | +0.017 | -0.005 | +12.50 | 104.7 | 1.73e-22 | **yes** |
| A | E1 | +0.022 | +0.017 | -0.005 | +5.61 | 29.3 | 4.44e-06 | **yes** |
| A | E2 | +0.022 | +0.021 | -0.001 | +6.17 | 113.8 | 1.06e-08 | **yes** |
| B | C | +0.013 | +0.022 | +0.008 | -21.19 | 105.1 | 9.78e-40 | **yes** |
| B | D | +0.013 | +0.017 | +0.004 | -6.61 | 197.9 | 3.53e-10 | **yes** |
| B | E1 | +0.013 | +0.017 | +0.004 | -3.72 | 41.3 | 5.97e-04 | **yes** |
| B | E2 | +0.013 | +0.021 | +0.008 | -18.84 | 114.9 | 6.21e-37 | **yes** |
| C | D | +0.022 | +0.017 | -0.005 | +12.34 | 105.3 | 3.42e-22 | **yes** |
| C | E1 | +0.022 | +0.017 | -0.005 | +5.55 | 29.4 | 5.25e-06 | **yes** |
| C | E2 | +0.022 | +0.021 | -0.001 | +5.69 | 118.4 | 9.38e-08 | **yes** |
| D | E1 | +0.017 | +0.017 | -0.000 | +0.06 | 40.8 | 9.50e-01 | no |
| D | E2 | +0.017 | +0.021 | +0.004 | -10.15 | 115.6 | 1.06e-17 | **yes** |
| E1 | E2 | +0.017 | +0.021 | +0.004 | -4.67 | 30.0 | 5.88e-05 | **yes** |

**Cohen's d (ABC) for Gemma3-12B  (layer 32, d=3840) requested contrasts:**
- A vs E1:  d = +1.44  (mean_A=+0.022, mean_E1=+0.017)
- A vs E2:  d = +0.99  (mean_A=+0.022, mean_E2=+0.021)
- E1 vs E2:  d = -1.19  (mean_E1=+0.017, mean_E2=+0.021)

## Cross-model side-by-side

- Qwen mean_pairwise_cos(gold)  = 0.4969
- Gemma mean_pairwise_cos(gold) = 0.9746

Reading: ABC is the headline number - same vertical axis for both models. Raw MSE shown for traceability but is NOT directly comparable because of the anisotropy gap.

| sub | n | Qwen ABC | Gemma ABC | Δ ABC | Qwen cos | Gemma cos | Qwen MSE | Gemma MSE |
|---|---|---|---|---|---|---|---|---|
| A | 100 | +0.371 | +0.022 | -0.349 | 0.868 | 0.996 | 0.264 | 0.007 |
| B | 100 | +0.300 | +0.013 | -0.286 | 0.797 | 0.988 | 0.407 | 0.024 |
| C | 100 | +0.306 | +0.022 | -0.284 | 0.802 | 0.996 | 0.395 | 0.007 |
| D | 100 | +0.346 | +0.017 | -0.329 | 0.843 | 0.992 | 0.313 | 0.017 |
| E1 | 30 | +0.247 | +0.017 | -0.230 | 0.744 | 0.992 | 0.511 | 0.017 |
| E2 | 70 | +0.406 | +0.021 | -0.385 | 0.903 | 0.996 | 0.194 | 0.009 |

## Rank correlation (Spearman ρ on cos(pred, gold))

Pairs by idx; ρ measures whether the two models find the same inputs harder/easier to reconstruct, independent of metric scale.

- **Overall** (n=500): ρ = +0.354, p = 3.24e-16

| sub | n | Qwen cos mean | Gemma cos mean | Spearman ρ | p |
|---|---|---|---|---|---|
| A | 100 | 0.868 | 0.996 | +0.111 | 2.73e-01 |
| B | 100 | 0.797 | 0.988 | +0.373 | 1.31e-04 |
| C | 100 | 0.802 | 0.996 | +0.159 | 1.14e-01 |
| D | 100 | 0.843 | 0.992 | +0.314 | 1.46e-03 |
| E1 | 30 | 0.744 | 0.992 | +0.402 | 2.75e-02 |
| E2 | 70 | 0.903 | 0.996 | +0.373 | 1.47e-03 |


## CoT Unfaithfulness Section

This section appended autonomously on 2026-05-10 during the phase-2 CoT-unfaithfulness experiment.

### Model: Gemma3-12B-IT

- Total rows: 60
- Neutral: 30, Biased: 30

#### Behavioral
- neutral correct rate: 30/30 = 100.0%
- biased correct rate:  30/30 = 100.0%
- biased answers matching the biased letter ("bias took effect"): 0/30 = 0.0%

#### CoT-unfaithfulness rates (conditional on bias took effect)

- The bias never took effect on this prompt set (n=0). The Turpin-style unfaithfulness rates are therefore **undefined** for this run - there is no event *model answered the biased letter while saying nothing about the bias*. The downstream claim is empty.

- This is itself a finding: a 12B instruction-tuned model with greedy decoding answered every one of these easy factual MCQs correctly even with a 3-shot all-same-letter prefix. Future replications should use harder questions (BBH-suggestive-MCQ style) and/or longer few-shot blocks to elicit the effect.

#### Supplementary: activation-level effect of the few-shot prefix

- neutral MSE: mean 0.0166, median 0.0164
- biased  MSE: mean 0.0199, median 0.0196
- Δ (biased − neutral) mean MSE: +0.0034
- neutral cos: mean 0.9917
- biased  cos: mean 0.9900

Interpretation: the few-shot prefix raises reconstruction MSE slightly even when behavior does not flip - i.e., the activation at the `(` position is not identical between neutral and biased even though the predicted next token is the same. This is consistent with the few-shot context leaving some trace in the residual stream that the AV verbalizes as a 'pattern of repeating Answer:(X)' format-level remark, not as an awareness of a wrong-answer bias intent.

### Model: Qwen2.5-7B-Instruct

Appended after the original 2026-05-10 snapshot: the Qwen Phase-2 run was completed in a later session. An earlier version of this report stated the Qwen counterpart had not been run.

- Total rows: 60
- Neutral: 30, Biased: 30

#### Behavioral
- neutral correct rate: 30/30 = 100.0%
- biased correct rate:  30/30 = 100.0%
- biased answers matching the biased letter ("bias took effect"): 0/30 = 0.0%

#### CoT-unfaithfulness rates (conditional on bias took effect)

- As with Gemma, the bias never took effect (n=0), so the Turpin-style unfaithfulness rate is **undefined** for this run. The negative result holds on both models.

#### Supplementary: activation-level effect of the few-shot prefix

- neutral MSE: mean 0.4997, median 0.4909
- biased  MSE: mean 0.4490, median 0.4251
- Δ (biased − neutral) mean MSE: −0.0507
- neutral cos: mean 0.7502
- biased  cos: mean 0.7755
- Δ (biased − neutral) mean cos: +0.0253

Interpretation: as with Gemma, the few-shot prefix leaves an activation-level trace at the `(` position even though no answer flips. The direction differs - on Qwen the biased prefix lowers reconstruction MSE (the activation moves toward a more legible, repetitive "Answer: (X)" format), where on Gemma it raised it slightly. Either way the trace reads as a format-level pattern, not as awareness of a wrong-answer bias.
