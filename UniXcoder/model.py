import torch
import torch.nn as nn
import torch
import itertools
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import torch.nn.functional as F

class UniXcoderModel(nn.Module):
    def __init__(self, encoder, config, args):
        super(UniXcoderModel, self).__init__()
        self.encoder = encoder 
        self.config = config 
        self.args = args   
        self.query = 0   

        # self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs_embeds=None, input_ids=None, labels=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(1)
        if inputs_embeds is not None:
            logits = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
        else:
            logits = self.encoder(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=input_ids.ne(1))[0]
        # prob = self.softmax(logits)
        prob = F.softmax(logits, dim=-1)
        input_size = input_ids.size() if input_ids is not None else inputs_embeds.size()
        assert prob.size(0) == input_size[0], (prob.size(), input_size)
        
        assert prob.size(1) == 2, prob.size()
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, prob
        else:
            return prob, logits

    def get_results(self, dataset, batch_size):

        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(input_ids=inputs, labels=label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = [[prob[0], 1 - prob[0]] for prob in logits]
        pred_labels = [0 if label else 1 for label in logits[:, 0] > 0.5]
        return probs, pred_labels