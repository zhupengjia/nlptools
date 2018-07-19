#!/usr/bin/env python

import torch,  sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..encoders.fconv_encoder import FConvEncoder
from ..encoders.fconv_decoder import FConvDecoder
from ..modules.model_base import ModelBase
from ..modules.bucket import BucketData


class FConvSeq2Seq(ModelBase):
    def __init__(self, encoder_vocab, decoder_vocab, encoder_layers=((256, 3),)*4, decoder_layers=((256, 3),)*3, decoder_attention=True,  dropout=0.1, max_source_positions=1024, max_target_positions=1024, share_input_output_embed=False, normalization_constant=0.5, device='cpu'):
        super().__init__()
   
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        
        self.device = torch.device(device)

        self.encoder = FConvEncoder(
            vocab = self.encoder_vocab,
            convolutions=encoder_layers,
            dropout=dropout,
            max_positions=max_source_positions,
            normalization_constant=normalization_constant,
        )
        
        self.decoder = FConvDecoder(
            vocab = self.decoder_vocab,
            out_embed_dim=self.decoder_vocab.embedding_dim,
            convolutions=decoder_layers,
            attention=decoder_attention,
            dropout=dropout,
            max_positions=max_target_positions,
            share_embed=share_input_output_embed,
            normalization_constant=normalization_constant,
        )
        
        self.encoder.num_attention_layers = sum(layer is not None for layer in self.decoder.attention)

    
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out
    
    
    def max_positions(self):
        """Maximum length supported by the model."""
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

                #print('batch_in', batch_in)
                #print('batch_prev', batch_prev)
                #print('batch_out', batch_out)
                #sys.exit()

                tag_scores = self.get_normalized_probs(self(batch_in, batch_len, batch_prev), log_probs=True)
                
                tag_scores_flatten = tag_scores.view(-1, self.decoder_vocab.vocab_size)
                targets_flatten = batch_out.view(-1)
                
                loss = loss_function(tag_scores_flatten, targets_flatten)
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0: 
            print('Epoch loss {}'.format(loss))
            torch.save(self.state_dict(), save_path)


