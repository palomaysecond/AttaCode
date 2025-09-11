import os
import torch
import numpy as np
from tqdm import tqdm
import json
import logging
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn
logger = logging.getLogger(__name__)


class CodeBERTInputFeatures(object):  
    
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

    return CodeBERTInputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))


class CodeBERTTextDataset(Dataset):  
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

    eval_dataset = CodeBERTTextDataset(tokenizer, args, args.test_data_file)
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
    logits = []
    labels = []
    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_for_lookup = {example.idx: example.input_tokens for example in eval_dataset.examples}

    for batch in tqdm(eval_dataloader, desc="Evaluating and computing attributions"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        input_ids.requires_grad_(False)
        label1 = label
        label_for_gather = label1.unsqueeze(1)
        batch_indices = batch[2].cpu().numpy()

        embedding = embedding_layer(input_ids)
        embedding.requires_grad_(True)
        attention_mask = input_ids.ne(1)

        logits_raw, _ = model(inputs_embeds=embedding, attention_mask=attention_mask)

        if model.config.num_labels == 1:

            prob_of_correct_class = torch.sigmoid(logits_raw).squeeze(-1)
        else:

            prob_softmax = torch.softmax(logits_raw, dim=-1)
            prob_of_correct_class = torch.gather(prob_softmax, 1, label_for_gather).squeeze()

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

        logits.append(logits_raw.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    with open(os.path.join(args.output_dir, "tokens.json"), 'w', encoding='utf-8') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.output_dir, "token_grad_norms.npz"), **grads_by_idx_to_save)

    return True



def predict_vulnerability(code, model, tokenizer, args):

    js = {"func": code, "idx": 0, "target": 0}  # 标签无关紧要，因为我们只关心预测结果
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
    feature = codebert_convert_examples_to_features(js, tokenizer, args)

    dataset = []
    dataset.append((torch.tensor([feature.input_ids]), torch.tensor(feature.label), torch.tensor(feature.idx)))

    model.eval()

    inputs = torch.tensor([feature.input_ids]).to(args.device)

    prob, logit = model(input_ids=inputs)

    is_vulnerable = logit > 0

    prob_diff = prob - (1 - prob)


    return is_vulnerable, logit, prob