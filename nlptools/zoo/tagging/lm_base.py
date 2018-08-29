#!/usr/bin/env python

import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nlptools.utils import eval_str_list
from .model_base import ModelBase
from ..encoders.fconv_decoder import FConvDecoder
from ..modules.bucket import BucketData

class LanguageModelBase(ModelBase):
    def __init__(self, vocab, pretrained_embed=True, device='cpu'):
        super().__init__()
        self.pretrained_embed = pretrained_embed 
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.device = torch.device(device)

    
    def train(self, inputs, targets, num_epoch=20, max_words = 100, save_path = 'autosave.torch'):
        # NLL is equivalent to the multi-category cross-entropy.
        loss_function = nn.NLLLoss(ignore_index=self.vocab.PAD_ID)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
    
        for epoch in range(num_epoch):
            #if epoch % 10 == 0: 
            print('Starting epoch {}'.format(epoch))
                
            buckets = BucketData([inputs, targets], max_words = max_words)
            for (batch_inputs, batch_tags) ,batch_lengths, in buckets:
                self.zero_grad()
              
                batch_inputs = torch.LongTensor(batch_inputs, device=self.device)
                batch_tags = torch.LongTensor(batch_tags, device=self.device)

                #print('batch_inputs', batch_inputs)
                #print('tags', batch_tags)
                
                tag_scores = self.get_normalized_probs(self(batch_inputs), log_probs=True)
                
                tag_scores_flatten = tag_scores.view(-1, self.vocab_size)
                targets_flatten = batch_tags.view(-1)
                
                loss = loss_function(tag_scores_flatten, targets_flatten)
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0: 
            print('Epoch loss {}'.format(loss))
            torch.save(self.state_dict(), save_path)


