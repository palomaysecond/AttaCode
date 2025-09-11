import os

model_path = "../unixcoder-base"
data_dir = "../../dataset/Devign"#"../../dataset/BigVul""../../dataset/Devign""../../dataset/ReVeal"
output_dir = "../saved_models_Devign"
command = f"""
CUDA_VISIBLE_DEVICES=1 python run.py \
    --output_dir={output_dir} \
    --model_name_or_path {model_path} \
    --do_test \
    --train_data_file={data_dir}/train.jsonl \
    --eval_data_file={data_dir}/valid.jsonl \
    --test_data_file={data_dir}/test.jsonl \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 12345 2>&1| tee ../saved_models_Devign/test.log
"""
os.system(command)

