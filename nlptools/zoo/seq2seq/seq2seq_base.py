#!/usr/bin/env python

import torch
from ..tagging.model_base import ModelBase
import torch.nn as nn
import torch.optim as optim
from ..modules.bucket import BucketData


class Seq2SeqBase(ModelBase):
    def __init__(self, encoder_vocab, decoder_vocab=None, pretrained_embed=True, share_embed=False, decoder_share_embed=False, device='cpu'):
        super().__init__()

        self.encoder_vocab = encoder_vocab

        self.device = torch.device(device)

        if decoder_vocab is None:
            self.share_embed = True
        else:
            self.share_embed = share_embed

        if self.share_embed:
            self.decoder_vocab = encoder_vocab
            self.decoder_pretrained_embed = False
        else:
            self.decoder_vocab = decoder_vocab
            self.decoder_pretrained_embed = pretrained_embed


    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out
    
    
    def max_positions(self):
        return (self.encoder.max_positions(), self.decoder.max_positions())

      
    def train(self, inputs, prev_outputs, outputs, num_epoch=20, max_words = 100, save_path = 'autosave.torch'):
        # NLL is equivalent to the multi-category cross-entropy.
        loss_function = nn.NLLLoss(ignore_index=self.encoder_vocab.PAD_ID)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
    
        for epoch in range(num_epoch):
            #if epoch % 10 == 0: 
            print('Starting epoch {}'.format(epoch))

            buckets = BucketData([inputs, prev_outputs, outputs], max_words = max_words)
            for (batch_in, batch_prev, batch_out) ,batch_len in buckets:
                self.zero_grad()
              
                batch_in = torch.LongTensor(batch_in, device=self.device)
                batch_prev = torch.LongTensor(batch_prev, device=self.device)
                batch_out = torch.LongTensor(batch_out, device=self.device)

                #print('batch_in', batch_in.size())
                #print('batch_prev', batch_prev.size())
                print('batch_out', targets_flatten.size())
                #sys.exit()

                tag_scores = self.get_normalized_probs(self(batch_in, batch_len, batch_prev), log_probs=True)
                
                tag_scores_flatten = tag_scores.view(-1, self.decoder_vocab.vocab_size)
                targets_flatten = batch_out.view(-1)
               
                #print('pred_out', tag_scores_flatten.size())

                loss = loss_function(tag_scores_flatten, targets_flatten)
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0: 
            print('Epoch loss {}'.format(loss))
            torch.save(self.state_dict(), save_path)



        

