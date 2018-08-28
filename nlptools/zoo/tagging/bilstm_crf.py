#!/usr/bin/env python
import torch, logging
import torch.nn as nn
import torch.autograd as autograd

from torch.nn.utils.rnn import pad_sequence, pack_sequence, PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.INFO)

log_model = logging.getLogger('model')
log_model_decoder = logging.getLogger('model.decoder')
log_model.setLevel(logging.INFO)
log_model_decoder.setLevel(logging.INFO)

torch.manual_seed(1)


def argmax(batch_vec):
    # return the argmax as a python int
    # log_model.debug('Argmax for %s\n%s', batch_vec.shape, batch_vec)
    _, idx = torch.max(batch_vec, 1)
    # log_model.debug('get argmax: %s', idx)
    return idx


def log_sum_exp(batch_vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm.
    The trick is: $log(1 + x) \approx x$ when $0 < x << 1$. 
    So, we can have $\log \sum_i \exp{x_i} = \log (\exp{a} + \sum_i \exp(b_i)$.
    """
    batch_size = batch_vec.shape[0]
    
    max_score = batch_vec[range(batch_size), argmax(batch_vec)]
    max_score_broadcast = max_score.view(batch_size, -1)\
                            .expand(batch_size, batch_vec.size()[1])
    # sum over #tags, i.e., dim=1
    score_bias = torch.log(torch.sum(torch.exp(batch_vec - max_score_broadcast), dim=1))
    
    ans = max_score + score_bias
    return ans
        


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        """
        The transition matrix is of shape `[tagset_size * tagset_size]`, in which each row is the destination, each column is the source. 
        E.g., `transitions[i]` selects the scores that arrives at the tag index `i` from the last time step. 
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        log_model.debug('vocab_size: %d, hidden_dim: %d, embedding_dim: %d', 
                        vocab_size, hidden_dim // 2, embedding_dim)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, batch_first=False, 
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        
    def init_hidden(self, batch_size):
        """
        Inputs: input, (h_0, c_0)
            input of shape (seq_len, batch, input_size): 
                tensor containing the features of the input sequence. 
                The input can also be a packed variable length sequence. 
                See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
            h_0 of shape (num_layers * num_directions, batch, hidden_size): 
                tensor containing the initial hidden state for each element in the batch.
            c_0 of shape (num_layers * num_directions, batch, hidden_size): 
                tensor containing the initial cell state for each element in the batch.
        """
        return (torch.randn(2, batch_size, self.hidden_dim // 2),
                torch.randn(2, batch_size, self.hidden_dim // 2))

    
    def _forward_alg(self, feats):
        """
        Do the forward algorithm to compute the partition function.
        
        Args:
            feats: PackedSequence
            batch_size: number of sequences in the batch (not to be confused with `batch_sizes`), 
                    which we will name `step_batch_size`
        """
        log_model.debug('shape of feats: %s', feats[0].shape)
        batch_size = feats[1][0].item()
        init_alphas = torch.full((batch_size, self.tagset_size), 
                                 -10000.)
        log_model.debug('shape of alphas: %s', init_alphas.shape)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
#        log_model.debug('forward_var: %s\n%s', forward_var.shape, forward_var)

        batch_start_ix = 0
        for step_size in feats[1].tolist():
            log_model.debug('working on step_size: %d', step_size)
            batch_end_ix = batch_start_ix + step_size
            feat = feats[0][batch_start_ix : batch_end_ix]
            batch_start_ix = batch_end_ix
#            log_model.debug('got the feature shape %s', feat.shape)
            
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    step_size, -1).expand(step_size, self.tagset_size)
                #log_model.debug('emit score shape: %s', feat[:, next_tag].shape)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(
                    1, -1).expand(step_size, self.tagset_size)
                #log_model.debug('trans_score shape: %s', trans_score.shape)

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var[:step_size] + trans_score + emit_score
                
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(step_size))
            
            assert len(alphas_t) == self.tagset_size,\
                    'at each step, the len of `alphas_t` should be the same as the `tagset_size`'
            
            # update only part of the forward var according to the step_size
            log_model.debug('alpha_t before cat:')
            for x in alphas_t:
                log_model.debug(x)
            alphas_t = torch.stack(alphas_t).transpose(0, 1)
            log_model.debug('catted alphas_t: \t%s', alphas_t)
            forward_var[:step_size] = alphas_t
        
        log_model.debug('final transition score: %s', self.transitions[self.tag_to_ix[STOP_TAG]])
        log_model.debug('forward_var: %s', forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    
    def _get_lstm_features(self, X):
        """
        Running the LSTM RNN to generate the potentials for the (linear-)CRF. 
        
        Args:
            X: the input sequence(s). 
                In batch mode, the input is in time-major format,
                of shape [seq_len, batch_size, embedding_dim]
                
        Return:
            lstm_feats: generate the potential for each tag. 
                In batch mode, the return value is in time major format, 
                of shape [seq_len, batch_size, tagset_size]
        """
        
        log_model.debug('Will run LSTM on X, length: %d (%s)', len(X), type(X))
        # padded_X = pad_sequence(X, batch_first=False, padding_value=PAD_IX)
        # embeds = self.word_embeds(padded_X )
        
        #lengths = [len(x) for x in X]
        #log_model.debug('lengths: \n%s', lengths)
        #packed_embeds = pack_padded_sequence(embeds, lengths=lengths, batch_first=False)
        packed_X = pack_sequence(X)
        log_model.debug('Did get packed X: %s, %s', packed_X[0].shape, packed_X[1])
        packed_embeds = PackedSequence(self.word_embeds(packed_X[0]), packed_X[1])
        log_model.debug('will run LSTM on: %s, %s', packed_embeds[0].shape, packed_embeds[1])
        lstm_out, self.hidden = self.lstm(packed_embeds)
        #log_model.debug('Did get the lstm_out from LSTM: \n%s', lstm_out)
                
        # however, we need to use only the data part of the output, 
        lstm_feats = self.hidden2tag(lstm_out[0])
        # and re-construct the packed sequence object as the return value. 
        output_feats = PackedSequence(lstm_feats, packed_embeds[1])
        #log_model.debug('Will return from `_get_lstm_features` with return value shpe: %s',
        #                output_feats[0].shape)
        return output_feats

    
    def _score_sentence(self, feats, Y):
        """
        Gives the score of a provided tag sequence. 
        Inputs are expected to be in time-major format.
        
        Args:
            lstm_feats: generate the potential for each tag. 
                In batch mode, the return value is in time major format, 
                of shape [seq_len, batch_size, tagset_size]
        """
        
        batch_size = feats[1][0].item()
        packed_Y = pack_sequence(Y)

        # there should be a score for each instance (sequence) in the batch
        score = torch.zeros(batch_size)
        #log_model.debug('Init score shape: %s', score.shape)
         
        previous_tag = torch.tensor([self.tag_to_ix[START_TAG]], 
                                       dtype=torch.long).expand(batch_size)
        ending_tag = torch.zeros_like(previous_tag)
        batch_start_ix = 0
        for step_size in feats[1].tolist():
            batch_end_ix = batch_start_ix + step_size
            feat = feats[0][batch_start_ix : batch_end_ix]
            current_tag = packed_Y[0][batch_start_ix : batch_end_ix]
            batch_start_ix = batch_end_ix
            
            #log_model.debug('the tags of the previous timestep: %s', previous_tag)
            #log_model.debug('the tags of the current timestep: %s', current_tag)
            
            transition_score = self.transitions[current_tag, previous_tag[:step_size]]
            #log_model.debug('get transition score: %s', transition_score)
            
            #log_model.debug('feat shape of current timestep: %s', feat.shape)
            # need to expand an extra dimension to indexing into the batch dim
            emission_score = feat[range(step_size), current_tag]
            #log_model.debug('got emission score: %s', emission_score)
            
            score[:step_size] = score[:step_size] + transition_score + emission_score
            
            # finally, don't forget to set the new previous_tag
            ending_tag[:step_size] = current_tag
            previous_tag = current_tag
            #log_model.debug('ending tag: %s\n---', ending_tag)
        
        # add the final stop-tag score to the overall score
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], ending_tag]
        #log_model.debug('final score: %s', score)
        
        return score

    
    def _viterbi_decode(self, feats):
        
        log_model_decoder.debug('transition matrix: \n%s\n---', self.transitions)
        backpointers = []

        # Initialize the viterbi variables in log space
        batch_size = feats[1][0].item()

        init_vvars = torch.full((batch_size, self.tagset_size), -10000.)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        
        batch_start_ix = 0
        for step_size in feats[1].tolist():
            batch_end_ix = batch_start_ix + step_size
            feat = feats[0][batch_start_ix : batch_end_ix]
            batch_start_ix = batch_end_ix

            bptrs_t = torch.zeros((step_size, self.tagset_size), dtype=torch.long)  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                trans_score = self.transitions[next_tag].view(1, -1).expand(step_size, 
                                                                            self.tagset_size)
                #log_model_decoder.debug('trans_score shape: %s', trans_score.shape)
                #log_model_decoder.debug('forward_var shape: %s', forward_var.shape)
                next_tag_var = forward_var[:step_size] + trans_score
                #log_model_decoder.debug('next_tag_var shape: %s', next_tag_var.shape)
                
                best_tag_id = argmax(next_tag_var)
                #log_model_decoder.debug('TAG %d: shape of best_tag_id: %s', next_tag, best_tag_id)
                
                bptrs_t[:step_size, next_tag] = best_tag_id
                #log_model_decoder.debug('next_tag_var[:, best_tag_id] shape: %s', next_tag_var[:, best_tag_id].shape)
                viterbivars_t.append(next_tag_var[range(step_size), best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            #log_model_decoder.debug('emission score: %s\n%s\n---', feat.shape, feat)
            cat_viterbivars_t = torch.stack(viterbivars_t).transpose(0, 1)
            #log_model_decoder.debug('cat_viterbivars_t: %s\n%s\n---', 
            #                        cat_viterbivars_t.shape, 
            #                        cat_viterbivars_t)
            forward_var[:step_size] = (cat_viterbivars_t + feat)
            #log_model_decoder.debug('forward_var: %s\n%s\n---', forward_var.shape, forward_var)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        log_model_decoder.debug('termainal_var: %s', terminal_var)
        best_tag_id = argmax(terminal_var)
        log_model_decoder.debug('best tag of termainal_var: %s', best_tag_id)
        path_score = terminal_var[range(batch_size), best_tag_id]

        for step in reversed(backpointers):
            log_model_decoder.debug(step)
        log_model_decoder.debug('-' * 20)
        # Follow the back pointers to decode the best path.
        ending_tag = best_tag_id.clone()
        best_path = [[_best_tag_id.item()] for _best_tag_id in best_tag_id]
        for bptrs_t in reversed(backpointers):
            step_size = len(bptrs_t)
            exp_size = min(step_size, len(best_tag_id))
            
            log_model_decoder.debug('bptrs_t: \n%s', bptrs_t)
            log_model_decoder.debug('step size: %d, exp size: %d, best_tag_id: \n%s', 
                                    step_size, exp_size, best_tag_id)
            log_model_decoder.debug('state of current best path: \n%s\n---', best_path)
            
            ending_tag[:exp_size] = best_tag_id[:exp_size]
            best_tag_id = bptrs_t[range(step_size), ending_tag[:step_size]]
            for i in range(step_size):
                best_path[i].append(best_tag_id[i].item())
        # Pop off the start tag (we dont want to return that to the caller)
        start = [best_path_.pop() for best_path_ in best_path]
        #assert (start == self.tag_to_ix[START_TAG]).all()  # Sanity check
        for best_path_ in best_path: best_path_.reverse()
        
        return path_score, best_path

    
    def neg_log_likelihood(self, X, Y):
        """
        Calculate the loss value given an input sentence and target tag-sequence. 
        The loss value is given by the negative-log-likelihood, which is 
        $$ 
            - \log P(Y | X) = - \log \Pi_t P(y_t | x_{t:T}) \\
                = - \sum_t \log P(y_t | x_{1:T}) \\
                = - \sum_t \log \frac{ e^{\hat{y}_t} }{ \sum_{y_t^{(i)} \in \mathcal{Y}} e^{y_t^{(i)}} } \\
                = \sum_t \left( \log \sum_{y_t^{(i)} \in \mathcal{Y}} e^{y_t^{(i)}} - \log e^{\hat{y}_t} \right)
        $$
        
        Args:
            X, Y: input and target output batch of sequences, in Python native list. 
        """
        assert len(X) == len(Y), \
                'the length of X should be the same as the lengthof Y'

        batch_size = len(X)
        
        X_sorted = sorted(X, reverse=True, key=lambda x:len(x))
        Y_sorted = sorted(Y, reverse=True, key=lambda x:len(x))
        feats = self._get_lstm_features(X_sorted)
        #log_model.debug('got feats from lstm: %s', feats)
        
        forward_score = self._forward_alg(feats)
        log_model.debug('got forward score %s', forward_score)
        
        gold_score = self._score_sentence(feats, Y_sorted)
        log_model.debug('got gold score %s', gold_score)
        
        return forward_score - gold_score

    
    def forward(self, X):  
        """
        Get the emission scores from the BiLSTM.
        This function is not required for training, since the loss function is calculated by `neg_log_likelihood`. 
        Note: dont confuse this with `_forward_alg` above.
        
        Parameters:
            sentence - 
        
        Returns:
            score - The accumlated score (NLL) the entire sequence input
            tag_seq - the optimal decoded tag-sequence. 
            
        Raises:
        """
        
        X_sorted = sorted(X, reverse=True, key=lambda x:len(x))
        
        lstm_feats = self._get_lstm_features(X_sorted)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    

    
