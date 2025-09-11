import os
import torch
import javalang
from javalang.tree import Node
import numpy as np
from tqdm import tqdm
import json
import logging
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn
from CodeT5.run import codet5_convert_examples_to_features

logger = logging.getLogger(__name__)


class CodeBertInputFeatures(object):  
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens  
        self.input_ids = input_ids  
        self.idx=idx  
        self.label=label  

def codebert_convert_examples_to_features(js,tokenizer,args):  
    
    code=' '.join(js['func'].split())  
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]  

    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]  

    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)  

    padding_length = args.block_size - len(source_ids)  
    source_ids+=[tokenizer.pad_token_id]*padding_length  

    return CodeBertInputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))


class CodeBertTextDataset(Dataset):  
    def __init__(self, tokenizer, args, file_path=None):  
        self.examples = []  
        

        file_type = file_path.split('/')[-1].split('.')[0]  
        folder = '/'.join(file_path.split('/')[:-1])  
        cache_file_path = os.path.join(folder, 'codebert_cached_{}'.format(file_type))

        try:
            self.examples = torch.load(cache_file_path)
        except:
            with open(file_path) as f:
                for line in f:  
                    js = json.loads(line.strip())  
                    self.examples.append(codebert_convert_examples_to_features(js, tokenizer, args))  
            torch.save(self.examples, cache_file_path)

        if 'train' == file_type:  
            for idx, example in enumerate(self.examples[:3]):  
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    
    def __len__(self):  
        return len(self.examples)

    def __getitem__(self, i):  
        
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].idx)  


def evaluate(args, model, tokenizer):
    """评估模型在测试数据集上的性能，并计算梯度归因和L2范数"""
    eval_dataset = CodeBertTextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if len(eval_dataset) == 0:
        raise ValueError(f"Test dataset {args.test_data_file} is empty or failed to load.")

    active_path = 'encoder.roberta.embeddings.word_embeddings'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        print(f"使用嵌入层路径: {active_path}")
    except AttributeError:
        raise ValueError(f"无法找到嵌入层路径 {active_path}，请检查模型结构")

    # evaluation
    model.eval()
    logits = []
    labels = []


    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_for_lookup = {example.idx: example.input_tokens for example in eval_dataset.examples}

    # batch_num = 0
    # tokens_by_idx = {example.idx: example.input_tokens for example in eval_dataset.examples}

    for batch in tqdm(eval_dataloader, desc="Evaluating and computing attributions"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        input_ids.requires_grad_(False)

        batch_indices = batch[2].cpu().numpy()

        batch_tokens = [tokens_by_idx_for_lookup[idx] for idx in batch_indices]

        embedding = embedding_layer(input_ids)
        embedding.requires_grad_(True)

        # prob = model(input_ids=input_ids, inputs_embeds=embedding)
        # prob = prob[:, 0]
        output = model(input_ids=input_ids, inputs_embeds=embedding)
        if isinstance(output, tuple):
            batch_logits = output[0]
        else:
            batch_logits = output.logits  # ModelOutput

        prob = torch.sigmoid(batch_logits)  
        prob = prob[:, 0]

        prob_diff = prob - (1 - prob)
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
            serializable_tokens = [t.replace('\u0120', '_') for t in batch_tokens[i]]
            tokens_by_idx_to_save[str(original_idx)] = serializable_tokens
            grads_by_idx_to_save[str(original_idx)] = batch_token_grads[i]

        logits.append(prob.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits > 0.5
    eval_acc = np.mean(labels == preds)

    result = {
        "eval_acc": round(eval_acc, 4),
    }


    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for idx, (label, pred) in enumerate(zip(labels, preds)):
            f.write(f"{idx}\t{int(pred)}\t{int(label)}\n")


    with open(os.path.join(args.output_dir, "tokens.json"), 'w') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)


    np.savez(os.path.join(args.output_dir, "token_grad_norms.npz"), **grads_by_idx_to_save)

    return result

def predict_vulnerability(code, model, tokenizer, args):

  
    js = {"func": code, "idx": 0, "target": 0}  
    feature = codebert_convert_examples_to_features(js, tokenizer, args)


    dataset = []
    dataset.append((torch.tensor([feature.input_ids]), torch.tensor(feature.label), torch.tensor(feature.idx)))

  
    active_path = 'encoder.roberta.embeddings.word_embeddings'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        print(f"Using embedding layer path: {active_path}")
    except AttributeError:
        raise ValueError(f"Unable to find the embedding layer path {active_path}, please check the model structure.")


    model.eval()

    inputs = torch.tensor([feature.input_ids]).to(args.device)

    input_embeddings = embedding_layer(inputs)

    embedding_weights = embedding_layer.weight.detach().cpu().numpy()

    embedding_info = {
        "active_path": active_path,
        "embedding_shape": input_embeddings.shape,
        "vocab_size": embedding_weights.shape[0],
        "embedding_dim": embedding_weights.shape[1],
        "tokens": [tokenizer.convert_ids_to_tokens(token_id) for token_id in feature.input_ids]
    }
    nums = 0
    all_nums = 0
    pad_nums = 0
    for token_id in feature.input_ids:
        all_nums += 1
        if tokenizer.convert_ids_to_tokens(token_id) != '<pad>':
            nums += 1
        else:
            pad_nums += 1


    outputs = model(input_ids = inputs, inputs_embeds = input_embeddings)
    # prob1 = outputs.cpu().numpy()[0][0]
    prob = outputs[0][0]

    is_vulnerable = prob > 0.5

    prob_diff = prob - (1 - prob)
    print(type(prob_diff))
    print(type(input_embeddings))

    embedding = torch.tensor(input_embeddings, dtype=torch.float32, requires_grad=True)

    emb_grad = torch.autograd.grad(
    outputs=prob_diff,
    inputs=input_embeddings,
    retain_graph=True,
    allow_unused=True
)[0]

    token_l2 = torch.norm(emb_grad, p=2, dim=2)  

   
    non_zero = token_l2 != 0
    valid_grad = token_l2[non_zero]
    
    normed_grad = torch.zeros_like(token_l2)
    normed_grad[non_zero] = (valid_grad - valid_grad.min()) / (valid_grad.max() - valid_grad.min() + 1e-8)

    return is_vulnerable, prob, input_embeddings, embedding_info, emb_grad, token_l2, normed_grad


def vulnerability_detect(code, model, tokenizer, args):

    js = {"func": code, "idx": 0, "target": 0}

    if args.model_name == 'codet5':

        feature = codet5_convert_examples_to_features(js, tokenizer, args)
        input_ids = torch.tensor([feature.input_ids]).to(args.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        model.eval()
        with torch.no_grad():
            prob, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        is_vulnerable = (prob[:, 1] > 0.5).item()
        confidence = prob[:, 1].item()

        return is_vulnerable, logits, confidence

    else:

        feature = codebert_convert_examples_to_features(js, tokenizer, args)
        dataset = [(torch.tensor([feature.input_ids]), torch.tensor(feature.label), torch.tensor(feature.idx))]

        model.eval()
        inputs = torch.tensor([feature.input_ids]).to(args.device)

        prob, logits = model(input_ids=inputs)

        if args.model_name == 'codebert':
            is_vulnerable = logits > 0.5
        elif args.model_name == 'graphcodebert':
            no_vuln_logit = logits[0][0] 
            no_vuln_prob = torch.sigmoid(no_vuln_logit)
            is_vulnerable = (no_vuln_prob <= 0.5).item()
        elif args.model_name == 'unixcoder':
            vuln_logit = logits[0][1]
            is_vulnerable = (vuln_logit > 0.5).item()
        else:
            raise ValueError(f"Unsupported model_name: {args.model_name}")

        prob_diff = prob - (1 - prob)

        return is_vulnerable, logits, prob_diff


class CodeT5InputFeatures(object):
    """Feature for CodeT5 classification."""
    def __init__(self, tokens, input_ids, idx, label):
        self.tokens = tokens
        self.input_ids = input_ids
        self.idx = idx
        self.label = label
