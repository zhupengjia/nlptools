#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nlptools.text import Vocab
from ..modules.bucket import BucketData, prepare_sequence


def to_one_hot(inputs, num_class):
    one_hot = torch.zeros(inputs.shape + (num_class,))
    index = inputs.view(inputs.shape + (1,))
    expand_dim = len(index.shape) - 1
    one_hot.scatter_(expand_dim, index, 1.)
    return one_hot


class LSTMTagger(nn.Module):
    
    def __init__(self, 
                 embedding_dim, hidden_dim, 
                 vocab_size, tagset_size,
                 num_layers, batch_size=1, max_len=128, 
                 bucket_config=None, device=None):
        
        super(LSTMTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.bucket_config = bucket_config
        
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.max_len = max_len

        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
        
    def init_hidden(self):
        # the hidden state should have the shape of 
        #  (num_layer*num_dirction, batch, hid_dim*num_direction)
        # and don't forget the extra cell-state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
               torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
        

    def forward(self, sentence):
        """
            Args: 
                sentence: The input sentence, word idx not mapped into embedding space, in the shape of `(batch?, len)`
                `nn.Embedding` would accept any shape tensor with value type Long 
        """
        embeds = self.word_embeddings(sentence)
        # now make the len to be inferred
        lstm_out, self.hidden = self.lstm(
            embeds.view(self.batch_size, -1, self.embedding_dim), self.hidden)
        
        # generate the logits of the tags
        logits = lstm_out.view(self.batch_size, -1, self.embedding_dim)
        tag_space = self.hidden2tag(logits)

        # then squash into the probabiliy score
        # meaning of the dims: (len, batch, emb)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores
    
    
    def sequence_loss(self, inputs, targets):
        loss_function = nn.NLLLoss(ignore_index=Vocab.PAD_ID)
        targets_one_hot = to_one_hot(targets, self.tagset_size)
        outputs = self(inputs).view(targets_one_hot.shape)
        xent = outputs * targets_one_hot
        xent = xent.sum(dim=2, keepdim=False)
        return xent
            
    

    def train(self, inputs, targets, num_epoch=20, save_path='autosave.torch'):
        # NLL is equivalent to the multi-category cross-entropy.
        loss_function = nn.NLLLoss(ignore_index=Vocab.PAD_ID)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
    
        for epoch in range(num_epoch):
            #if epoch % 10 == 0: 
            print('Starting epoch {}'.format(epoch))
                
            buckets = BucketData(inputs, targets, self.bucket_config)
            for batch_inputs, batch_tags in buckets:
                self.zero_grad()
                self.hidden = self.init_hidden()
               
                batch_inputs = batch_inputs.to(self.device)
                batch_tags = batch_tags.to(self.device)

                print('batch_inputs', batch_inputs)
                print('tags', batch_tags.size())
                
                tag_scores = self(batch_inputs)
                
                tag_scores_flatten = tag_scores.view(-1, self.tagset_size)
                targets_flatten = batch_tags.view(-1)
                
                loss = loss_function(tag_scores_flatten, targets_flatten)
                loss.backward()
                optimizer.step()
                sys.exit()
            #if epoch % 10 == 0: 
            print('Epoch loss {}'.format(loss))
            torch.save(self.state_dict(), save_path)
                
                
    def load_params(self, save_path):
        self.load_state_dict(torch.load(save_path))



