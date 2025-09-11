import torch
import os
import time
from datetime import datetime
import argparse
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from model import CodeBERTModel
from run import CodeBERTTextDataset, predict_vulnerability, evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")  # roberta

    parser.add_argument("--tokenizer_name", default='/root/autodl-tmp/codebert-base', type=str,
                        help="Path to pretrained tokenizer (local directory or HuggingFace repo).")

    parser.add_argument("--model_name_or_path", default='/root/autodl-tmp/codebert-base', type=str,
                        help="Path to pretrained model (local directory or HuggingFace repo).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default="../dataset/Devignn/test1.jsonl", type=str,
                        help="Path to test data file.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store pretrained models.")

    parser.add_argument("--checkpoint_path", default="saved_models/checkpoint-best-acc/codebert_model.bin", type=str,
                        help="Path to fine-tuned checkpoint (.bin file)")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 1  

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=False,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = CodeBERTModel(model, config, tokenizer, args)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    print(f"[INFO] Loading fine-tuned weights from: {args.checkpoint_path}")

    state_dict = torch.load(args.checkpoint_path, map_location=args.device)

    for k in list(state_dict.keys()):
        if "classifier" in k:
            print(f"Skipping incompatible key: {k}")
            del state_dict[k]
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device), strict=False)
    model.to(args.device)

    result = evaluate(args, model, tokenizer)
    print("Evaluation result:", result)


if __name__ == '__main__':
    main()