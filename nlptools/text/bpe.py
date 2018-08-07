#!/usr/bin/env python
import copy, re, numpy, os, sys
from .vocab import Vocab
from collections import defaultdict


class BytePair(Vocab):
    '''
        Learn Byte Pair Encoding, see https://arxiv.org/abs/1508.07909 
    '''
    EOW = '</w>'
    EOW_ID = 4

    def __init__(self, bpe_size=1000, min_freq=2, separator='@@', code_file='', **args):
        '''
            Byte Pair Encoding (BPE) vocabulary for rare word, see https://arxiv.org/abs/1508.07909. Note the bpe vocab is included in total vocab

            Input:
                - min_freq: int, Stop if no symbol pair has frequency >= min_freq, default is 2
                - bpe_size: number of learned bpe symbols, default is 1000
                - separator: separator symbol for bpe split, default is '@@'
                - code_file: trained bpe code file, default is ''
                - any available parameters in nlptools.text.vocab
        '''
        self.bpe_size = bpe_size
        self.min_freq = min_freq
        self.separator = separator
        self.code_file = code_file
        super().__init__(**args)
        self._word_spec.append(self.EOW)
        self._id_spec.append(self.EOW_ID)
        self.load(code_file)


    def load(self, code_file):
        '''
            load bpe code from bpe ref file

            Input:
                - bpe_code: trained bpe code file, default is ''
        '''
        self.__init_codes()
        if os.path.exists(code_file):
            self.bpe_codes = []
            with open(code_file) as f:
                for l in f:
                    if l[0] == '#': continue
                    data = tuple(re.split('\s', l.strip()))
                    self.bpe_codes.append(data)
        self.__reverse_bpe_codes()


    def __init_codes(self):
        self.bpe_codes, self.bpe_codes_reverse = {}, {} 


    def _get_cached_vocab(self):
        ifinit = True
        if os.path.exists(self.cached_vocab):
            try:
                self.bpe_cache = zload(self.cached_vocab)['bpe_cache']
                if len(self.bpe_cache) > 0:
                    ifinit = False
            except Exception as err:
                print(err)
        if ifinit: 
            self.bpe_cache = {}
        super(BytePair, self)._get_cached_vocab()


    def save(self):
        '''
            Save the vocab dictionary to *cached_vocab*
        '''
        if len(self.cached_vocab) > 0:
            zdump({'word2id':self._word2id, 'id2tf':self._id2tf, 'bpe_cache':self.bpe_cache}, self.cached_vocab)


    def learn(self, refresh=True):
        '''
            learn bpe

            Input:
                - refresh: bool, check if add to existed bpe codes, if True will create a new one, default is True
        '''
        if refresh:
            self.__init_codes()

        word2tf = {}
        for w in self._word2id:
            if w not in self._word_spec and len(w) > 1:
                word2tf[w] = self._id2tf[self._word2id[w]]
        word2tf = dict([(tuple(x[:-1])+(x[-1]+self.EOW,) ,y) for (x,y) in word2tf.items()])
        sorted_word2tf = sorted(word2tf.items(), key=lambda x: x[1], reverse=True)

        stats, indices = self.__get_pair_statistics(sorted_word2tf)
    
        big_stats = copy.deepcopy(stats)

        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(stats.values()) / 10

        self.bpe_codes = []

        if len(self.code_file) > 0:
            outfile = open(self.code_file, 'w')
        else:
            outfile = None

        
        for i in range(self.bpe_size):
            if stats:
                most_frequent = max(stats, key=lambda x: (stats[x], x))

            # we probably missed the best pair because of pruning; go back to full statistics
            if not stats or (i and stats[most_frequent] < threshold):
                self.__prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: (stats[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i/(i+10000.0)
                self.__prune_stats(stats, big_stats, threshold)

            if stats[most_frequent] < self.min_freq:
                break

            if outfile:
                outfile.write('{0} {1}\n'.format(*most_frequent))
            
            self.bpe_codes.append(most_frequent)
            changes = self.__replace_pair(most_frequent, sorted_word2tf, indices)
            
            self.__update_pair_statistics(most_frequent, changes, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                self.__prune_stats(stats, big_stats, threshold)
        self.__reverse_bpe_codes()

        if outfile:
            outfile.close()
        #print(self.bpe_codes, len(self.bpe_codes))
        #print(self.bpe_codes_reverse)
        

    def __reverse_bpe_codes(self):
        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])
    

    
    def __replace_pair(self, pair, word2tf, indices):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        first, second = pair
        pair_str = ''.join(pair)
        pair_str = pair_str.replace('\\','\\\\')
        changes = []
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
        for j, freq in indices[pair].items():
            if freq < 1:
                continue
            word, freq = word2tf[j]
            new_word = ' '.join(word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(' '))
    
            word2tf[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))
    
        return changes


    def __prune_stats(self, stats, big_stats, threshold):
        """Prune statistics dict for efficiency of max()
    
        The frequency of a symbol pair never increases, so pruning is generally safe
        (until we the most frequent pair is less frequent than a pair we previously pruned)
        big_stats keeps full statistics for when we need to access pruned items
        """
        for item,freq in list(stats.items()):
            if freq < threshold:
                del stats[item]
                if freq < 0:
                    big_stats[item] += freq
                else:
                    big_stats[item] = freq


    def __update_pair_statistics(self, pair, changed, stats, indices):
        """Minimally update the indices and frequency of symbol pairs
    
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        stats[pair] = 0
        indices[pair] = defaultdict(int)
        first, second = pair
        new_pair = first+second
        for j, word, old_word, freq in changed:
    
            # find all instances of pair, and update frequency/indices around it
            i = 0
            while True:
                # find first symbol
                try:
                    i = old_word.index(first, i)
                except ValueError:
                    break
                # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
                if i < len(old_word)-1 and old_word[i+1] == second:
                    # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                    if i:
                        prev = old_word[i-1:i+1]
                        stats[prev] -= freq
                        indices[prev][j] -= 1
                    if i < len(old_word)-2:
                        # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                        # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                        if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                            nex = old_word[i+1:i+3]
                            stats[nex] -= freq
                            indices[nex][j] -= 1
                    i += 2
                else:
                    i += 1
    
            i = 0
            while True:
                try:
                    # find new pair
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
                if i:
                    prev = word[i-1:i+1]
                    stats[prev] += freq
                    indices[prev][j] += 1
                # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
                # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
                if i < len(word)-1 and word[i+1] != new_pair:
                    nex = word[i:i+2]
                    stats[nex] += freq
                    indices[nex][j] += 1
                i += 1


    def __get_pair_statistics(self, word2tf):
        """Count frequency of all symbol pairs, and create index"""
    
        # data structure of pair frequencies
        stats = defaultdict(int)
    
        #index from pairs to words
        indices = defaultdict(lambda: defaultdict(int))
    
        for i, (word, freq) in enumerate(word2tf):
            prev_char = word[0]
            for char in word[1:]:
                stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                prev_char = char
    
        return stats, indices
 

    def __recursive_split(self, segment, final=False):
        """Recursively split segment into smaller units (by reversing BPE merges)
        until all units are either in-vocabulary, or cannot be split futher."""
    
        try:
            if final:
                left, right = self.bpe_codes[segment + self.BPE]
                right = right[:-4]
            else:
                left, right = self.bpe_codes[segment]
        except:
            #sys.stderr.write('cannot split {0} further.\n'.format(segment))
            yield segment
            return
    
        if left + self.separator in vocab:
            yield left
        else:
            for item in self.__recursive_split(left, False):
                yield item
    
        if (final and right in vocab) or (not final and right + self.separator in vocab):
            yield right
        else:
            for item in self.__recursive_split(right, final):
                yield item

    def __check_vocab_and_split(self, orig):
        """Check for each segment in word if it is in-vocabulary,
        and segment OOV segments into smaller units by reversing the BPE merge operations"""
    
        out = []
    
        for segment in orig[:-1]:
            if segment + self.separator in self._word2id:
                out.append(segment)
            else:
                #sys.stderr.write('OOV: {0}\n'.format(segment))
                for item in self.__recursive_split(segment, False):
                    out.append(item)
        segment = orig[-1]
        if segment in self._word2id:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in self.__recursive_split(segment, True):
                out.append(item)
        
        return out

    def __get_pairs(self, word):
        """Return set of symbol pairs in a word.
    
        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs


    def __encode(self, orig):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """
    
    
        if orig in self.bpe_cache:
            return self.bpe_cache[orig]

        if len(self.bpe_codes) < 1:
            return [orig]

        word = tuple(orig[:-1]) + ( orig[-1] + self.EOW,)

        pairs = self.__get_pairs(word)

        if not pairs:
            return [orig]

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_codes.get(pair, float('inf')))
            
            if bigram not in self.bpe_codes:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
            
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.__get_pairs(word)

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>',''),)

        word = self.__check_vocab_and_split(word)

        self.bpe_cache[orig] = word
        return word


    def word2id(self, word):
        '''
            Convert word to id, with apply of bpe

            Input: 
                - word: string
            
            Output:
                - wordid: int    
        '''
        if word is None: return None
        if word in self._word_spec or len(word) <2:
            new_words = [word]
        else:
            new_words = self.__encode(word)
            for i in range(len(new_words)-1):
                if new_words[i][-len(self.separator):] != self.separator:
                    new_words[i] += self.separator
        #calculate number of bpe words
        return [super(BytePair, self).word2id(w) for w in new_words]
        

    def words2id(self, tokens, batch=False):
        '''
            tokens to token ids, with apply of bpe

            Input:
                - tokens: list of token
                - batch: if the input sequence is a batches, default is False
            
            Output:
                - list of ids
        '''
        if batch:
            return numpy.asarray([self.apply(t) for t in tokens], dtype=numpy.object) 
        
        ids = []
        for token in tokens:
            i = self.word2id(token)
            if i is not None:
                ids += self.word2id(token)
        ids = numpy.array(ids, 'int')
        
        return ids


