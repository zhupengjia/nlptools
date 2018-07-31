#!/usr/bin/env python3
"""DrQA Document Reader model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from torch.autograd import Variable
from .rnn_reader import RnnDocReader

from nlptools.utils import setLogger


class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, vocab, model_type='rnn', feature_dict=None, max_len = 1000,
                 state_dict=None, normalize=True, fix_embedding=True, optimizer='sgd', learning_rate=0.001, momentum=0, weight_decay=1e-5, grad_clipping=0, tune_partial=0, **args):
        # Book-keeping.
        self.logger = setLogger()
        self.vocab = vocab
        
        self.feature_dict = feature_dict
        self.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.fix_embedding = fix_embedding
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clipping = 0
        self.tune_partial = 0
        self.max_len = max_len

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if model_type == 'rnn':
            self.network = RnnDocReader(vocab=vocab, num_features=self.num_features, normalize=normalize, **args)
        else:
            raise RuntimeError('Unsupported model: %s' % self.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.vocab.word2id(w) for w in words
                  if w not in self.vocab}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            self.logger.info('Adding %d new words to dictionary...' % len(to_add))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.vocab.vocab_hash_size,
                                                        self.vocab.emb_ins.vec_len,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """
            Load pretrained embeddings for a given list of words, if they exist.

            Args:
                - words: iterable of tokens. Only those that are indexed in the dictionary are kept.
                - embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        self.logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = parsed[0]
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        self.logger.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        self.logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            self.logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            self.logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        elif self.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """
            Forward a batch of examples only to get predictions.

            Args:
                - ex: the batch
                - candidates: batch * variable length list of string answer options.
                  The model will only consider exact spans contained in this list.
                - top_n: Number of predictions to return per batch element.
                - async_pool: If provided, non-gpu post-processing will be offloaded
                  to this CPU process pool.

            Output:
                - pred_s: batch * top_n predicted start indices
                - pred_e: batch * top_n predicted end indices
                - pred_score: batch * top_n prediction scores

            If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'feature_dict': self.feature_dict,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            self.logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'feature_dict': self.feature_dict,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            self.logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename):
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']

        saved_params = {'args': args, 'feature_dict':feature_dict, 'state_dict':state_dict}

        return saved_params
        

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']

        model = DocReader(args, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
