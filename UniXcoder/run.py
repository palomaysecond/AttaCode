from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pandas as pd
import random
import json
import re
import torch
import numpy as np
import pickle

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import tqdm
from tqdm import tqdm

from model import *
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import sys

logger = logging.getLogger(__name__)

class UniXcoderInputFeatures(object):


    def __init__(self, input_tokens, input_ids, index, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def unixcoder_convert_examples_to_features(js, tokenizer, args):

    code = ' '.join(js['func'].split())

    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]

    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return UniXcoderInputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))

class UniXcoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        m = re.search(r"(train|valid|test)", file_path)

        if m is None:  
            partition = None
        else:
            partition = m.group(1)  

        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(unixcoder_convert_examples_to_features(js, tokenizer, args))



        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        # self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(self.examples[i].index)  
        )

def evaluate(args, model, tokenizer):
    eval_dataset = UniXcoderTextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    active_path = 'encoder.roberta.embeddings.word_embeddings'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        print(f"Using embedding layer path: {active_path}")
    except AttributeError:
        raise ValueError(f"Unable to find the embedding layer path {active_path}, please check the model structure.")

    # 评估
    model.eval()
    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_for_lookup = {example.index: example.input_tokens for example in eval_dataset.examples}

    for batch in tqdm(eval_dataloader, desc="Evaluating and computing attributions for UniXcoder"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        batch_indices = batch[2].cpu().numpy()

        input_ids.requires_grad_(False)
        label_for_gather = label.unsqueeze(1)
        # batch_tokens = [tokens_by_idx_for_lookup[idx] for idx in batch_indices]

        embedding = embedding_layer(input_ids)
        embedding.requires_grad_(True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        prob, _ = model(inputs_embeds=embedding, attention_mask=attention_mask)

        prob_of_correct_class = torch.gather(prob, 1, label_for_gather).squeeze()

        prob_diff = prob_of_correct_class - (1 - prob_of_correct_class)

        batch_token_grads = []
        for i in range(input_ids.size(0)):
            grad_i = torch.autograd.grad(
                outputs=prob_diff[i],
                inputs=embedding,
                retain_graph=True
            )[0][i]
            token_l2 = torch.norm(grad_i, p=2, dim=1)

            non_zero = token_l2 != 0
            valid_grad = token_l2[non_zero]

            normed_grad = torch.zeros_like(token_l2)
            if torch.numel(valid_grad) > 0 and valid_grad.max() > valid_grad.min():
                normed_grad[non_zero] = (valid_grad - valid_grad.min()) / (valid_grad.max() - valid_grad.min() + 1e-8)

            batch_token_grads.append(normed_grad.detach().cpu().numpy())

        for i, original_idx in enumerate(batch_indices):
            serializable_tokens = [str(t) for t in tokens_by_idx_for_lookup[original_idx]]
            tokens_by_idx_to_save[str(original_idx)] = serializable_tokens
            grads_by_idx_to_save[str(original_idx)] = batch_token_grads[i]

    tokens_output_path = os.path.join(args.output_dir, "tokens.json")
    with open(tokens_output_path, 'w', encoding='utf-8') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)

    grads_output_path = os.path.join(args.output_dir, "token_grad_norms.npz")
    np.savez(grads_output_path, **grads_by_idx_to_save)


    return True