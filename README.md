

# AttaCode: Attribution-Guided Adversarial Code Generation via Jailbreak-Enabled Prompts against Vulnerability Detection Models

This repository provides the implementation of training, analyzing, and attacking pre-trained code models such as **CodeBERT**, **UniXcoder**, and **CodeT5**.

---

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourname/yourrepo.git
   cd yourrepo

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

------

## Training Pretrained Models

- For **CodeBERT** and **UniXcoder**, go to the corresponding training folder and run:

	```bash
	python train.py
	```

- For **CodeT5**, we follow the official implementation:
	  [CodeT5 GitHub Repository](https://github.com/salesforce/CodeT5)

------

## Code Processing and Importance Analysis

After training the model, process the dataset and compute token importance scores.

### Example: Running `CodeProcess.py`

```bash
python CodeProcess.py \
  --model_name_or_path ./models/codebert-base \
  --tokenizer_name ./models/codebert-base \
  --checkpoint_path ./CodeBERT/saved_models/Devign/checkpoint-best-acc/model.bin \
  --test_data_file ./dataset/Devign/test.jsonl \
  --output_dir ./CodeBERT/saved_models/Devign \
  --block_size 512 \
  --eval_batch_size 4 \
  --seed 42
```

### Example: Running `ImportanceAnalyze.py`

```bash
python ImportanceAnalyze_codebert.py \
  --source_codes_path ./dataset/Devign/test.jsonl \
  --tokens_list_path ./CodeBERT/saved_models/Devign/tokens.json \
  --tokens_scores_path ./CodeBERT/saved_models/Devign/token_grad_norms.npz \
  --top_n 10 \
  --language c \
  --saved_filename ./CodeBERT/saved_models/Devign/transformation_importance_scores.json
```

------

## Iterative Attack

Before running iterative attacks, **fill in your API keys** in the `config.json` file.
 Example `config.json`:

```python
    "api_keys": {
        "gpt": "your api-key",
        "claude": "your api-key", 
        "deepseek": "your api-key",
        "qwen": "your api-key",
        "gemini": "your api-key"
    },
```

Then, perform iterative adversarial attacks on the fine-tuned model using external LLMs.

Finally, perform iterative adversarial attacks on the fine-tuned model using external LLMs.

### Example: Running `iterative_attack.py`

```bash
python iterative_attack.py \
  --model_type roberta \
  --model_name unixcoder \
  --tokenizer_name ./models/unixcoder-base \
  --model_name_or_path ./models/unixcoder-base \
  --output_dir ./UniXcoder/saved_models/BigVul \
  --eval_data_file ./dataset/BigVul/test.jsonl \
  --importance_score_file ./UniXcoder/saved_models/BigVul/transformation_importance_scores.json \
  --target_llm deepseek \
  --language c \
  --block_size 512 \
  --eval_batch_size 4 \
  --seed 42 \
  --result_file ./UniXcoder/saved_models/BigVul/attack_results_deepseek.jsonl \
  --finetuned_model_path ./UniXcoder/saved_models/BigVul/checkpoint-best-acc/model.bin \
  --prompt_log_file ./logs/bigvul_prompt.jsonl
```

##  Notes

- Modify paths in the example commands according to your environment.

- Ensure that `requirements.txt` is installed before running any script.

	

