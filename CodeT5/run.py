import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration  
from tqdm import tqdm
import argparse
import logging

from CodeT5.model import CodeT5Model
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CodeT5InputFeatures(object):
    def __init__(self, tokens, input_ids, idx, label):
        self.tokens = tokens
        self.input_ids = input_ids
        self.idx = idx
        self.label = label


def codet5_convert_examples_to_features(js, tokenizer, args):

    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.block_size - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length

    return CodeT5InputFeatures(tokens, input_ids, js['idx'], int(js['target']))


class CodeT5TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, f'codet5_cached_{file_type}')

        if os.path.exists(cache_file_path):
            logger.info(f"Loading cached dataset from {cache_file_path}")
            self.examples = torch.load(cache_file_path)
        else:
            logger.info(f"Creating features from {file_path}")
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    js = json.loads(line.strip())
                    self.examples.append(codet5_convert_examples_to_features(js, tokenizer, args))
            torch.save(self.examples, cache_file_path)

        if 'train' in file_type:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info(f"idx: {example.idx}")
                logger.info(f"label: {example.label}")
                logger.info(f"input_tokens: {example.tokens}")
                logger.info(f"input_ids: {' '.join(map(str, example.input_ids))}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feature = self.examples[idx]
        return (
            torch.tensor(feature.input_ids),
            torch.tensor(feature.label),
            torch.tensor(feature.idx)
        )


def evaluate(args, model, tokenizer):

    dataset = CodeT5TextDataset(tokenizer, args, args.test_data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    active_path = 'encoder.shared'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        logger.info(f"Using embedding layer path: {active_path}")
    except AttributeError:
        raise ValueError(f"Embedding layer path {active_path} not found in model.")

    model.eval()
    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_lookup = {ex.idx: ex.tokens for ex in dataset.examples}

    for batch in tqdm(dataloader, desc="Evaluating and computing attributions"):
        input_ids = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        indices = batch[2].cpu().numpy()

        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad_(True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        probs, _ = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        prob_of_correct = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
        prob_diff = prob_of_correct - (1 - prob_of_correct)

        batch_token_grads = []
        for i in range(input_ids.size(0)):
            grad_i = torch.autograd.grad(
                outputs=prob_diff[i],
                inputs=embeddings,
                retain_graph=True
            )[0][i]
            token_l2 = torch.norm(grad_i, p=2, dim=1)

            non_zero = token_l2 != 0
            valid_grad = token_l2[non_zero]
            normed_grad = torch.zeros_like(token_l2)
            if valid_grad.numel() > 0:
                normed_grad[non_zero] = (valid_grad - valid_grad.min()) / (valid_grad.max() - valid_grad.min() + 1e-8)

            batch_token_grads.append(normed_grad.cpu().numpy())

        for i, idx in enumerate(indices):
            tokens_by_idx_to_save[str(idx)] = tokens_by_idx_lookup[idx]
            grads_by_idx_to_save[str(idx)] = batch_token_grads[i]

    os.makedirs(args.output_dir, exist_ok=True)
    tokens_file = os.path.join(args.output_dir, "tokens.json")
    grads_file = os.path.join(args.output_dir, "token_grad_norms.npz")
    with open(tokens_file, 'w', encoding='utf-8') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)
    np.savez(grads_file, **grads_by_idx_to_save)

    logger.info(f"Saved tokens to {tokens_file}")
    logger.info(f"Saved gradient norms to {grads_file}")

    return True

def vulnerability_detect(code, model, tokenizer, args):

    js = {"func": code, "idx": 0, "target": 0}
    feature = codet5_convert_examples_to_features(js, tokenizer, args)

    input_ids = torch.tensor([feature.input_ids]).to(args.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    model.eval()
    with torch.no_grad():
        prob, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    is_vulnerable = (prob[:, 1] > 0.5).item()
    confidence = prob[:, 1].item()
    return is_vulnerable, confidence

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="codet5", type=str, help="Model type")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str, help="Pretrained tokenizer name")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Path to fine-tuned model")
    parser.add_argument("--test_data_file", required=True, type=str, help="Path to test dataset (jsonl)")
    parser.add_argument("--output_dir", required=True, type=str, help="Where to store output files")
    parser.add_argument("--block_size", default=512, type=int, help="Input block size")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)  
    config = T5Config.from_pretrained(args.model_name_or_path)
    encoder = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = CodeT5Model(encoder=encoder, config=config, tokenizer=tokenizer, args=args)
    model.to(args.device)

    logger.info("Starting evaluation...")
    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
