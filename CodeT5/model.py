import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from transformers import RobertaTokenizer  

class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder  
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0  
        self.classifier = nn.Linear(config.d_model, 2)  
        self.loss_func = nn.CrossEntropyLoss()

    def get_t5_vec(self, source_ids):

        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        encoder_outputs = self.encoder.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        vec = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return vec


    def forward(self, input_ids=None, labels=None, inputs_embeds=None, attention_mask=None):

        if input_ids is not None:

            total_size = input_ids.numel()
            block_size = self.args.block_size
            new_size = (total_size // block_size) * block_size
            if total_size != new_size:
                print(f"[WARNING] Trimming input_ids from {total_size} to {new_size} for block_size={block_size}")
                input_ids = input_ids.view(-1)[:new_size]

            input_ids = input_ids.view(-1, block_size)

            input_ids = input_ids.view(-1, self.args.block_size)
            vec = self.get_t5_vec(input_ids)
        elif inputs_embeds is not None:
            decoder_input_ids = torch.full(
                (inputs_embeds.size(0), 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=inputs_embeds.device
            )  

            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.decoder_hidden_states[-1]
            vec = hidden_states[:, -1, :]  
        else:
            raise ValueError("forward() requires either input_ids or inputs_embeds")

        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss, prob
        else:
            return prob, logits

    def get_results(self, dataset, batch_size):

        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4, pin_memory=False)

        self.eval()
        probs_all = []
        preds_all = []

        for batch in eval_dataloader:
            input_ids = batch[0].to(self.args.device)
            with torch.no_grad():
                prob, _ = self.forward(input_ids=input_ids)
                probs_all.extend(prob.cpu().numpy())
                preds_all.extend((prob[:, 1] > 0.5).long().cpu().numpy())

        return probs_all, preds_all
