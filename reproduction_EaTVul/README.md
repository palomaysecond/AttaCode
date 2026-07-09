# EaTVul Reproduction Code

This directory contains our reproduction of EaTVul, implemented according to the EaTVul paper and its released resources. The code runs the attack on vulnerability-detection models over the Devign, BigVul, and DiverseVul datasets.

## Structure

```text
reproduction_EaTVul/
  configs/eatvul_release.yaml
  eatvul_repro/
  scripts/
  requirements.txt
```

## Requirements

Create an environment with Python 3.10 or a compatible version, then install:

```powershell
python -m pip install -r requirements.txt
```

The LLM generator is called through AIHubMix. Set the API key through an environment variable:

```powershell
$env:AIHUBMIX_API_KEY = "your-key"
```

No API key is stored in the code.

## External Files

Before running the full experiments, edit `configs/eatvul_release.yaml` and provide:

1. dataset split paths for Devign, BigVul, and DiverseVul;
2. target checkpoint paths for CodeBERT, UniXcoder, and CodeT5;
3. the desired output directory.

## GPT Model

The original EaTVul implementation uses `gpt-3.5-turbo` as the ChatGPT-based generator. Because that backend has been superseded by newer ChatGPT-family models, this reproduction uses `gpt-4o`. All LLM calls are routed through AIHubMix, following the API service used in our experiments.

## Main Workflow

Run one dataset-model pair step by step:

```powershell
python .\scripts\prepare_attack_samples.py --config .\configs\eatvul_release.yaml --dataset Devign --target-model CodeBERT
python .\scripts\train_surrogate.py --config .\configs\eatvul_release.yaml --dataset Devign --target-model CodeBERT
python .\scripts\generate_attack_pool.py --config .\configs\eatvul_release.yaml --dataset Devign --target-model CodeBERT
python .\scripts\run_eatvul_attack.py --config .\configs\eatvul_release.yaml --dataset Devign --target-model CodeBERT
python .\scripts\evaluate_results.py --config .\configs\eatvul_release.yaml
```

For sequential runs:

```powershell
python .\scripts\run_public_target_batch.py --config .\configs\eatvul_release.yaml --pairs Devign:CodeBERT BigVul:CodeBERT DiverseVul:CodeBERT --resume
```

Add other pairs such as `Devign:UniXcoder` or `Devign:CodeT5` after their checkpoint paths are configured.

## Outputs

Runtime outputs are written under `outputs/eatvul/` by default:

```text
prepared/      originally correct vulnerable samples
surrogates/    BiLSTM-attention surrogate checkpoints
attack_pool/   generated and validated snippets
runs/          per-sample attack traces
summary/       ASR and AMQ summaries
cache/         LLM prompt/response cache
audit/         provider-error contamination audit
```

## Parameter Choices

Some implementation details required adaptation because the original EaTVul artifacts do not fully specify every parameter for this evaluation setting.

- Surrogate model: BiLSTM-attention surrogate trained per dataset and target detector.
- Knowledge distillation: target-model probabilities are used when available, with `kd_temperature = 5.0`, `kd_alpha = 0.9`, and `kd_beta = 0.1`.
- Feature selection: the top 10 attention-ranked tokens are used, with a 5-line context window.
- Generation: one snippet is requested per prompt, with at most four regeneration attempts.
- Validation: snippets must be code-like, avoid provider-error text and URLs, and use the `eatvul_` prefix for generated identifiers.
- FGA search: three clusters, population size 20, five generations, and at most four inserted snippets.
- Query stopping: the attack stops early once the target model flips from vulnerable to non-vulnerable.

All values can be changed in `configs/eatvul_release.yaml`.

## Result Audit

After a run, audit the outputs for provider-error contamination:

```powershell
python .\scripts\audit_outputs.py --output-dir .\outputs\eatvul --fail-on-errors
```

Provider quota or filtering messages should not be admitted into the final attack pool.
