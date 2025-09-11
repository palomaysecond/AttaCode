import os

output_dir = "../saved_models_DiverseVul"
data_dir = "../../dataset/DiverseVul"#"../../dataset/BigVul""../../dataset/Devign""../../dataset/ReVeal""../../dataset/DiverseVul"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_path = "microsoft/codebert-base"

commend = f"""
python run.py \
    --output_dir={output_dir} \
    --model_type=roberta \
    --tokenizer_name={model_path} \
    --model_name_or_path={model_path} \
    --do_train \
    --train_data_file={data_dir}/train.jsonl \
    --eval_data_file={data_dir}/valid.jsonl \
    --test_data_file={data_dir}/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee {output_dir}/train.log
"""
os.system(commend)