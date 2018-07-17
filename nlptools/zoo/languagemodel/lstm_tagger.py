#!/usr/bin/env python

import torch, sys
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..modules.bucket import BucketData

class LSTMTagger(nn.Module):
    
    def __init__(self, vocab, hidden_dim, tagset_size, num_layers = 1, dropout = 0, device='cpu'):
        super(LSTMTagger, self).__init__()

        self.vocab_size = vocab.vocab_size
        self.padding_idx = vocab.PAD_ID
        self.embedding_dim = vocab.embedding_dim
        self.num_layers = num_layers
        self.vocab = vocab
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, self.padding_idx)
        
        self.device = torch.device(device)

        self.tagset_size = tagset_size

        self.embedding.weight.data = torch.FloatTensor(vocab.dense_vectors()).to(self.device)


        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, dropout = dropout, batch_first=True) 

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    
    
    def forward(self, sentence, lengths):
        """
            Args: 
                sentence: The input sentence, word idx not mapped into embedding space, in the shape of `(batch?, len)`
                `nn.Embedding` would accept any shape tensor with value type Long 
        """
        embeds = self.embedding(sentence)

        embeds = pack_padded_sequence(embeds, lengths,  batch_first=True)

        # now make the len to be inferred
        lstm_out, self.hidden = self.lstm(
            embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out) 
        
        tag_space = self.hidden2tag(lstm_out)

        tag_score = F.log_softmax(tag_space, dim=2)


        return tag_score


    def train(self, inputs, targets, num_epoch=20, max_words = 100, save_path = 'autosave.torch'):
        # NLL is equivalent to the multi-category cross-entropy.
        loss_function = nn.NLLLoss(ignore_index=self.vocab.PAD_ID)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
    
        for epoch in range(num_epoch):
            #if epoch % 10 == 0: 
            print('Starting epoch {}'.format(epoch))
                
            buckets = BucketData(inputs, targets, max_words = max_words)
            for batch_inputs, batch_tags ,batch_lengths, in buckets:
                self.zero_grad()
              
                batch_inputs = torch.LongTensor(batch_inputs, device=self.device)
                batch_tags = torch.LongTensor(batch_tags, device=self.device)

                #print('batch_inputs', batch_inputs.size())
                #print('tags', batch_tags.size())
                
                tag_scores = self(batch_inputs, batch_lengths)
                    
                tag_scores_flatten = tag_scores.view(-1, self.tagset_size)
                targets_flatten = batch_tags.view(-1)
                
                loss = loss_function(tag_scores_flatten, targets_flatten)
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0: 
            print('Epoch loss {}'.format(loss))
            torch.save(self.state_dict(), save_path)


    def load_params(self, save_path):
        self.load_state_dict(torch.load(save_path))


