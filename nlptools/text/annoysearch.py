#!/usr/bin/env python
import os

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class AnnoySearch:
    '''
        Search keyword using annoy, via word vectors, please check `spotify/annoy <https://github.com/spotify/annoy>`_ for more details

        Input:
            - embedding: text.embedding object, for word2vec source
            - tokenizer: text.tokenizer object, for tokenizer
            - annoy_ntree: int, builds a forest of annoy_ntree trees, default is 10
            - annoy_cache: string, cached index file path, default is ''
            - annoy_filter: float, similarity threshold, default is 0.5
    '''
    def __init__(self, embedding, tokenizer = None, annoy_cache = '', annoy_ntree=10, annoy_filter=0.5):
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.annoy_cache = annoy_cache
        self.annoy_ntree = annoy_ntree
        self.annoy_filter = annoy_filter


    def load_index(self, keywords):
        '''
            Build or load index(if annoy_cache existed)

            Input:
                - keywords: list of string
        '''
        from annoy import AnnoyIndex
        self.search = AnnoyIndex(self.embedding.dim)
        if os.path.exists(self.annoy_cache):
            self.search.load(self.annoy_cache)
        else:
            self.keywords = keywords
            for i,k in enumerate(keywords):
                self.search.add_item(i,self.embedding[k])
            self.search.build(self.annoy_ntree)
            if len(self.annoy_cache) > 0:
                try:
                    self.search.save(self.annoy_cache)
                except Exception as err: 
                    print('Annoy cache failed! ' + err)


    def find(self, sentence, remove_stopwords=False, location=False):
        '''
            Search keywords from sentence

            Input:
                - sentence: string
                - remove_stopwords: bool, if remove stopwords or not, default is False
                - location: bool, if return keyword location or not, default is False

            Output:
                - list, format like [(keyword, location), ...] 

        '''
        if isinstance(sentence, str):
            sentence_seg = self.tokenizer.seg(sentence, remove_stopwords=remove_stopwords)['tokens']
        else:
            sentence_seg = sentence
        match = []
        for i, s in enumerate(sentence_seg):
            result = self.search.get_nns_by_vector(self.embedding[s], 1, include_distances=True)
            keyword, similarity = result[0][0], 1/(1+result[1][0])
            if similarity < self.annoy_filter:
                continue
            if location:
                if isinstance(sentence, str):
                    loc = sentence.index(s)
                else:
                    loc = i
                match.append((self.keywords[keyword], loc))
            else:
                match.append(self.keywords[keyword])
        return match
            

     
        
