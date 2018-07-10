#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch import autograd
from core.globals import globalvars as gvars


def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w], seq))
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    

class BiLSTM_CRF(nn.Module):
    
    def __init__(self):
        super(BiLSTM_CRF, self).__init__()
        self.cfg = gvars.cfg['ner']
        self.embedding_dim = self.cfg['embedding_dim']
        self.hidden_dim = self.cfg['hidden_dim']
        self.vocab_size = self.cfg['vocab_size']
        self.tagset_size = self.cfg['n_entities']
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, int(self.hidden_dim/2), num_layers=1, bidirectional=True)
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        # These two statements enforce the constraint that we never transfer *to* the start tag,
        # and we never transfer *from* the stop tag (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.start_tag = self.cfg['vocab']['START_TAG']
        self.stop_tag = self.cfg['vocab']['STOP_TAG']
        self.transitions.data[self.start_tag, :] = -10000
        self.transitions.data[:, self.stop_tag] = -10000

        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return ( autograd.Variable( torch.randn(2, 1, self.hidden_dim)),
                 autograd.Variable( torch.randn(2, 1, self.hidden_dim)) )
    
    
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.start_tag] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        
        # Iterate through the sentence
        for feat in feats:
            alphas_t = [] # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.stop_tag]
        alpha = log_sum_exp(terminal_var)
        return alpha
        
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
        
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable( torch.Tensor([0]) )
        tags = torch.cat( [torch.LongTensor([self.start_tag]), tags] )
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.stop_tag, tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        
        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.start_tag] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1 
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step
            
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.stop_tag]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.start_tag # Sanity check
        best_path.reverse()
        return path_score, best_path
        
    def neg_log_likelihood(self, sentence, tags):
        self.hidden = self.init_hidden()
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
        
    def forward(self, sentence): # dont confuse this with _forward_alg above.
        self.hidden = self.init_hidden()
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

