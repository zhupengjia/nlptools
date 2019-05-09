#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''
import numpy, math, torch
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine


def format_sentence(sentence, vocab, tokenizer=None, max_seq_len=50):
    """
        Format token ids to sentence and masks

        Input:
            - sentence: string or list of token ids, 
            - vocab:  instance of nlptools.text.vocab
            - tokenizer:  instance of nlptools.text.tokenizer, default is None
            - max_seq_len: int, default is 50
    """
    if tokenizer is None:
        token_ids = sentence
    else:
        tokens = tokenizer(sentence)
        if len(tokens) < 1:
            return None
        token_ids = vocab.words2id(tokens)[:max_seq_len-2]
    seq_len = len(token_ids) + 2
    sentence = numpy.zeros(max_seq_len, 'int')
    sentence_mask = numpy.zeros(max_seq_len, 'int')
    sentence[0] = vocab.BOS_ID
    sentence[1:seq_len-1] = token_ids
    sentence[seq_len-1] = vocab.EOS_ID
    sentence_mask[:seq_len] = 1
    return sentence, sentence_mask


class SimBase:
    def __init__(self, **args):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass

    def distance(self, vec1, vec2):
        '''
            Cosine distance between two vectors

            Input:
                - vec1: first vector
                - vec2: second vector
        '''
        return cosine(vec1, vec2)

class WMDSim(SimBase):
    '''
        Calculate similarities between sentences

        Input:
            - vocab: text.vocab object
    '''
    def __init__(self, vocab, **args):
        super(WMDSim, self).__init__(**args)
        self.vocab = vocab


    def min_word_distance(self, sentence_id1, sentence_id2):
        '''
            Minimum word distance between two sentences

            Input:
                - sentence_id1: word_id list for sentence1
                - sentence_id2: word_id list for sentence2 
        '''
        vec1 = self.vocab.ids2vec(sentence_id1)
        vec2 = self.vocab.ids2vec(sentence_id2)
        return cosine_distances(vec1, vec2).min()
    
   
    def wcd_distance(self, sentence_id1, sentence_id2):
        '''
            Word center distance between two sentences

            Input:
                - sentence_id1: word_id list for sentence1
                - sentence_id2: word_id list for sentence2 
        '''
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = numpy.unique(numpy.concatenate((sentence_id1, sentence_id2)))
        vectors = self.vocab.ids2vec(words)
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = numpy.zeros(len_words, 'float32')
            for d in doc:
                v[word2id[d]] += 1./max(self.vocab._id2tf[d], 1)
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)
        v1 = numpy.dot(d1, vectors)
        v2 = numpy.dot(d2, vectors)
        return self.distance(v1, v2)


    def wmd_distance(self, sentence_id1, sentence_id2):
        '''
            Word mover's distance between two sentences

            Input:
                - sentence_id1: word_id list for sentence1
                - sentence_id2: word_id list for sentence2 
        '''
        from pyemd import emd
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = numpy.unique(numpy.concatenate((sentence_id1, sentence_id2)))
        vectors = self.vocab.ids2vec(words)
        distance_matrix = cosine_distances(vectors)
        if numpy.sum(distance_matrix) == 0.0:
            return float('inf')
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = numpy.zeros(len_words, 'float32')
            for d in doc:
                v[word2id[d]] += 1./max(self.vocab._id2tf[d], 1)
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)
        return emd(d1, d2, distance_matrix)
    

    def rwmd_distance(self, sentence_id1, sentence_id2):
        '''
            Relaxation word mover's distance between two sentences

            Input:
                - sentence_id1: word_id list for sentence1
                - sentence_id2: word_id list for sentence2 
        '''
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = numpy.unique(numpy.concatenate((sentence_id1, sentence_id2)))
        vectors = self.vocab.ids2vec(words)
        distance_matrix = cosine_distances(vectors)
        if numpy.sum(distance_matrix) == 0.0:
            return 0
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = numpy.zeros(len_words, 'float')
            for d in doc:
                v[word2id[d]] += 1./max(self.vocab._id2tf[d], 1)
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)
        new_weights_dj = distance_matrix[:len(d1), d2>0].min(axis=1)
        new_weights_di = distance_matrix[:len(d2), d1>0].min(axis=1)

        rwmd = max(numpy.dot(new_weights_dj, d1), numpy.dot(new_weights_di, d2))
        return rwmd


    def idf_weighted_distance(self, sentence_id1, sentence_id2):
        '''
            IDF weighted distance between two sentences

            Input:
                - sentence_id1: word_id list for sentence1
                - sentence_id2: word_id list for sentence2 
        '''
        vec1 = self.vocab.ave_vec(sentence_id1)
        vec2 = self.vocab.ave_vec(sentence_id2)
        return self.distance(vec1, vec2)


class BERTSim(SimBase):
    '''
        Extract sentence embedding from bert

        Input:
            - encoder: BERT encoder
    '''
    def __init__(self, encoder, **args):
        super(BERTSim, self).__init__(**args)
        self.encoder = encoder
    
    def eval(self):
        self.encoder.eval()

    def to(self, device):
        self.encoder.to(device)
        self.device = device

    @property
    def dim(self):
        '''
            dimention of sentence embedding
        '''
        return self.encoder.config.hidden_size

    def get_embedding(self, sentences, sentence_masks):
        '''
            return sentence embedding

            Input:
                - sentences: torch tensor of sentence
                - sentence_masks: torch tensor of sentence masks
        '''
        sequence_output, pooled_output = self.encoder(sentences,
                                                      attention_mask=sentence_masks,
                                                      output_all_encoded_layers=False)
        return pooled_output.cpu().detach().numpy()

    def __call__(self, sentences, sentence_masks, batch_size=500):
        '''
            return sentence embedding

            Input:
                - sentences: list of sentence ids
                - sentence_masks: list of sentence masks
        '''
        sentences = numpy.stack(sentences, axis=0)
        sentence_masks = numpy.stack(sentence_masks, axis=0)
        embeddings = numpy.zeros((sentences.shape[0], self.dim))
        for i in range(math.ceil(sentences.shape[0]/batch_size)):
            starti = batch_size * i
            endi = min(batch_size * (i+1), sentences.shape[0])
            ids = torch.LongTensor(sentences[starti:endi]).to(self.device)
            masks = torch.LongTensor(sentence_masks[starti:endi]).to(self.device)
            embeddings[starti:endi] = self.get_embedding(ids, masks)
        return embeddings


