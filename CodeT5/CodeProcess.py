import torch
import os
import json
import argparse
from transformers import (
    T5Config, T5ForConditionalGeneration, RobertaTokenizer  
)
from model import CodeT5Model
from run import CodeT5TextDataset, vulnerability_detect, evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="codet5", type=str,
                        help="The model architecture to be used (codet5 / codebert / unixcoder).")
    parser.add_argument("--language", required=True, type=str,
                        help="Programming language of the dataset (java / python / c / cpp).")
    parser.add_argument("--tokenizer_name", required=True, type=str,
                        help="Path to pretrained tokenizer directory.")
    parser.add_argument("--model_name_or_path", required=True, type=str,
                        help="Path to pretrained model directory.")
    parser.add_argument("--checkpoint_path", required=True, type=str,
                        help="Path to fine-tuned checkpoint (model.bin).")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="Directory where predictions and outputs will be saved.")
    parser.add_argument("--test_data_file", required=True, type=str,
                        help="Path to test data file (JSONL).")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory for caching pretrained models.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f" Target programming language: {args.language}")

    print(f" Loading tokenizer from: {args.tokenizer_name}")
    tokenizer = RobertaTokenizer.from_pretrained(  
        args.tokenizer_name,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    print(f" Loading base model from: {args.model_name_or_path}")
    config = T5Config.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    encoder = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = CodeT5Model(encoder, config, tokenizer, args)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"[INFO] Loading fine-tuned weights from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)

    model.to(args.device)

    print(f"[INFO] Evaluating on: {args.test_data_file}")
    success = evaluate(args, model, tokenizer)  
    if success:
        print("[RESULT] Evaluation completed successfully.")
    else:
        print("[RESULT] Evaluation failed.")

if __name__ == '__main__':
    main()
