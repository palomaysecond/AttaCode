import json
import argparse
from tqdm import tqdm

import torch

from run import vulnerability_detect  
from your_model_loader import load_model_and_tokenizer  

def detect_all(input_file, output_file, model, tokenizer, args):

    results = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc="Processing samples"):
            data = json.loads(line)
            code = data['func']
            idx = data.get('idx', None)

            try:
                is_vuln, logit, prob = vulnerability_detect(code, model, tokenizer, args)

                result = {
                    "idx": idx,
                    "project": data.get("project"),
                    "commit_id": data.get("commit_id"),
                    "is_vulnerable": bool(is_vuln),
                    "logit": logit.detach().cpu().numpy().tolist() if torch.is_tensor(logit) else logit,
                    "prob": prob.detach().cpu().numpy().tolist() if torch.is_tensor(prob) else prob
                }
            except Exception as e:
                print(f"Error processing idx={idx}: {e}")
                result = {
                    "idx": idx,
                    "project": data.get("project"),
                    "commit_id": data.get("commit_id"),
                    "error": str(e)
                }

            results.append(result)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=2)
    print(f"\nDetection results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch vulnerability detection on test.jsonl")
    parser.add_argument("--input_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save detection results")
    parser.add_argument("--model_name", type=str, required=True, help="Model name: codebert, graphcodebert, unixcoder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path, args.tokenizer_path, args.device)

    detect_all(args.input_file, args.output_file, model, tokenizer, args)

if __name__ == "__main__":
    main()
