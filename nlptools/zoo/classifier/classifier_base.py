#!/usr/bin/env python

import torch
import torch.nn as nn

class ClassifierBase(nn.Module):
    def __init__(self, vocab, pretrained_embed=True, device='cpu'):
        super().__init__()
        self.vocab = vocab
        self.device = torch.device(device)

    
    def train(self, inputs, outputs, num_epoch=20, max_words=100, learning_rate=0.001, weight_decay=0, save_path='autosave.torch'):
        loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), \
                lr=learning_rate, \
                weight_decay=weight_decay)
        
        for epoch in range(num_epoch):
            print('starting epoch {}'.format(epoch))
            
            buckets = BucketData([inputs, outputs], max_words = max_words)

            for (batch_in, batch_prev, batch_out) ,batch_len in buckets:
                self.zero_grad()
              
                batch_in = torch.LongTensor(batch_in, device=self.device)
                batch_out = torch.LongTensor(batch_out, device=self.device)
                
                out_prob = self(batch_in)
                loss = loss_fn(out_prob, batch_out)

                _, out_pred = torch.max(out_prob.data, 1)
                precision = out_pred.eq(batch_out.data).sum()/batch_out.numel()

                print('{} {} {}'.format(epoch, loss.data[0], precision))

                loss.backward()
                optimizer.step()

        torch.save(self.state_dict(), save_path)
