import os

output_dir = "../saved_models_DiverseVul/"
data_dir = "../../dataset/DiverseVul"#"../../dataset/BigVul""../../dataset/Devign""../../dataset/ReVeal""../../dataset/DiverseVul"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_path = "../unixcoder-base"
commend = f"""
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    --output_dir {output_dir} \
    --model_name_or_path {model_path} \
    --do_train \
    --train_data_file={data_dir}/train.jsonl \
    --eval_data_file={data_dir}/valid.jsonl \
    --test_data_file={data_dir}/test.jsonl \
    --num_train_epochs 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 12345 2>&1 | tee ../saved_models_DiverseVul/train.log
"""


os.system(commend)

