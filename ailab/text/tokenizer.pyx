#!/usr/bin/env python
import os, string, numpy, re, glob, json, requests
from ..utils import restpost

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Segment_Base(object):
    '''
        Parent class for other tokenizer classes, please don't use this class directly

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - stopwords_path: the path of stopwords
                    - ner_name_replace: dictionary, replace the entity name to the mapped name
    '''
    def __init__(self, cfg):
        self.stopwords = {}
        self.cfg = {'stopwords_path':None, 'ner_name_replace':{}}
        for k in cfg: self.cfg[k] = cfg[k]
        self.__loadStopwords(self.cfg['stopwords_path'])

    def __loadStopwords(self, stopwords_path):
        if stopwords_path is not None and os.path.exists(stopwords_path):
            with open(stopwords_path, encoding='utf-8') as f:
                for i in f.readlines():
                    self.stopwords[i.strip()] = ''

    def __call__(self, sentence):
        return self.seg(sentence)


class Segment_CoreNLP(Segment_Base):
    '''
        Stanford CoreNLP wrapper, Please check `stanford CoreNLP <https://stanfordnlp.github.io/CoreNLP/>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - CORENLP_URL: corenlp api url
    '''
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)
        self.server_url = cfg['CORENLP_URL']

    def _annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        r = requests.post(self.server_url, params={'properties': str(properties)}, data=text.encode('utf-8'), headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
             and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output

    def tokensregex(self, text, pattern, filter):
        return self.regex('/tokensregex', text, pattern, filter)

    def semgrex(self, text, pattern, filter):
        return self.regex('/semgrex', text, pattern, filter)

    def regex(self, endpoint, text, pattern, filter):
        r = requests.get(
            self.server_url + endpoint, params={
                'pattern':  pattern,
                'filter': filter
            }, data=text)
        output = r.text
        try:
            output = json.loads(r.text)
        except:
            pass
        return output

    def seg(self, sentence, remove_stopwords = True, entities_filter = None, pos_filter = None):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True
                - entities_filter: string list, will only remain the entity tokens, default is None
                - pos_filter: string list, will only remain special pos tokens, default is None

            Output: dictionary with keys:
                - tokens: list of tokens
                - texts: list of raw texts
                - entities: list of entities from NER
                - pos: list of pos tags
        '''
        txts, tokens, entities, pos= [], [], [], []
        for idx, sentence in enumerate(self._annotate(sentence, properties={'annotators': 'tokenize, pos, lemma, ner', 'outputFormat':'json'})['sentences']):
            for token in sentence['tokens']:
                if remove_stopwords and token['word'] in self.stopwords:
                    continue
                if pos_filter is not None and token['pos'] not in pos_filter:
                    continue
                if self.cfg['ner_name_replace'] is not None and token['ner'] in self.cfg['ner_name_replace']:
                    token['ner'] = self.cfg['ner_name_replace'][token['ner']]
                if entities_filter is not None and token['ner'] not in entities_filter:
                    continue
                if len(token['lemma'])<1:
                    continue
                txts.append(token['word'])
                tokens.append(token['lemma'])
                pos.append(token['pos'])
                entities.append(token['ner'])
    
        return {"tokens":tokens, "texts":txts, "entities":entities, 'pos':pos}
   

class Segment_Spacy(Segment_Base):
    '''
        Spacy wrapper, Please check `Spacy <https://spacy.io/>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - cached_ner: if the key existed, will load the user defined model instead of default model.
    '''
    def __init__(self, cfg):
        import spacy
        Segment_Base.__init__(self, cfg)
        if 'cached_ner' in cfg:
            self.nlp = spacy.load(cfg['cached_ner'])
        else:
            self.nlp = spacy.load(cfg['LANGUAGE'])

    def seg(self, sentence, remove_stopwords = True, tags_filter = None, entities_filter = None, pos_filter = None, dep_filter=None):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True
                - tags_filter: string list, will only main special tag tokens, default is None
                - entities_filter: string list, will only remain the entity tokens, default is None
                - pos_filter: string list, will only remain special pos tokens, default is None
                - dep_filter: string list, will only remain special dep tokens, default is None

            Output: dictionary with keys:
                - tokens: list of tokens
                - tags: list of detailed part-of-speech tags
                - texts: list of raw texts
                - entities: list of entities from NER
                - pos: list of simple part-of-speech tags
                - dep: list of syntactic dependency
        '''
        txts, tokens, tags, entities, pos, dep= [], [], [], [], [], []
        for token in self.nlp(sentence):
            if remove_stopwords and token.text in self.stopwords:
                continue
            if tags_filter is not None and token.tag_ not in tags_filter:
                continue
            if pos_filter is not None and token.pos_ not in pos_filter:
                continue
            if dep_filter is not None and token.dep_ not in dep_filter:
                continue
            entity = token.ent_type_
            if self.cfg['ner_name_replace'] is not None and entity in self.cfg['ner_name_replace']:
                entitiy = self.cfg['ner_name_replace'][entity]
            if entities_filter is not None and entity not in entities_filter:
                continue
            if len(token.lemma_)<1:
                continue
            txt = token.text.strip()
            if len(txt) < 1 : continue
            txts.append(txt)
            tokens.append(token.lemma_)
            tags.append(token.tag_)
            entities.append(entity)
            pos.append(token.pos_)
            dep.append(token.dep_)
            
        return {"tokens":tokens, "tags":tags, "texts":txts, "entities":entities, 'pos':pos, 'dep':dep}
   

class Segment_Jieba(Segment_Base):
    '''
        Jieba wrapper, Please check `Jieba <https://github.com/fxsjy/jieba>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - seg_dict_path: if the key existed, will load the user definded model
    '''
    def __init__(self, cfg):
        import jieba
        Segment_Base.__init__(self, cfg)
        if 'seg_dict_path' in cfg:
            jieba.load_userdict(cfg['seg_dict_path'])
        self.nlp = jieba

    def seg(self, sentence, remove_stopwords=True):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True

            Output: dictionary with keys:
                - tokens: list of tokens
        '''
        #sentence = re.sub(r"[\s\u0020-\u007f\u2000-\u206f\u3000-\u303f\uff00-\uffef]+", " ", sentence)
        sentence = re.sub(r"[^\w\d]+", " ", sentence)
        tokens = []
        for x in self.nlp.cut(sentence, cut_all=False):
            x = x.strip()
            if remove_stopwords and x in self.stopwords:
                continue
            if len(x)<1:
                continue
            tokens.append(x)
        return {"tokens":tokens}
     

class Segment_LTP(Segment_Base):
    '''
        HIT-SCIR pyltp wrapper, Please check `pyltp <https://github.com/HIT-SCIR/pyltp>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - cws_model_path: path of cws model
                    - pos_model_path: path of pos model
                    - ner_model_path: path of ner model
                    - 
    '''
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)
        from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
        self.seg_ins = Segmentor()
        self.seg_ins.load(cfg['cws_model_path'])
        self.pos_ins = Postagger()
        self.pos_ins.load(cfg['pos_model_path'])
        if 'parser_model_path' in cfg and  os.path.exists(cfg['parser_model_path']):
            self.parser_ins = Parser()
            self.parser_ins.load(cfg['parser_model_path'])
        else:
            self.parser_ins = None
        self.ner_ins = []

        for path in sorted(glob.glob(cfg['ner_model_path'])):
            try:
                if os.path.getsize(path) > 1024: 
                    self.ner_ins.append(NamedEntityRecognizer())
                    self.ner_ins[-1].load(path)
            except Exception as err:
                print(err)

    def __del__(self):
        self.seg_ins.release()
        self.pos_ins.release()
        for n in self.ner_ins:
            n.release()

    def seg(self, sentence, remove_stopwords = True, tags_filter = None, entities_filter = None, entityjoin=True):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True
                - tags_filter: string list, will only main special tag tokens, default is None
                - entities_filter: string list, will only remain the entity tokens, default is None
                - entityjoin: bool, will join the tokens with same entity and next to each other, default is True

            Output: dictionary with keys:
                - tokens: list of tokens
                - tags: list of detailed part-of-speech tags
                - entities: list of entities from NER
        '''
        words_ = self.seg_ins.segment(sentence)
        postags_ = self.pos_ins.postag(words_)
        if self.parser_ins is not None:
            arcs_ = self.parser_ins.parse(words_, postags_)
        entities__ = []
        
        for n in self.ner_ins:
            entities__.append(list(n.recognize(words_, postags_)))
        #mix entities
        entities_ = entities__[0]
        for ee in entities__:
            for i,e in enumerate(ee):
                if e != 'O':
                    entities_[i] = e

        words_, postags_ = list(words_), list(postags_)
        words, postags, entities = [],[],[]
        word_tmp, postag_tmp, entity_tmp = '', [], []
        for i,w in enumerate(words_):
            if remove_stopwords and w in self.stopwords:
                continue
            entity = re.split('-', entities_[i])
            if len(entity) > 1:
                entity_loc, entity = entity[0], entity[1]
            else:
                entity_loc, entity = 'O', 'O'
            if self.cfg['ner_name_replace'] is not None and entity in self.cfg['ner_name_replace']:
                entity = self.cfg['ner_name_replace'][entity]
            if tags_filter is not None and postags_[i] not in tags_filter:
                continue
            if entities_filter is not None and entity not in entities_filter:
                continue
            if entityjoin:
                if entity_loc not in ['I', 'E'] and len(word_tmp) > 0:
                    words.append(word_tmp)
                    postags.append(postag_tmp[0])
                    entities.append(entity_tmp[0])
                    word_tmp, postag_tmp, entity_tmp = '', [], []
                if entity_loc in ['B', 'I', 'E']:
                    if len(entity_tmp) > 0 and entity != entity_tmp[-1]:
                        words.append(word_tmp)
                        postags.append(postag_tmp[0])
                        entities.append(entity_tmp[0])
                        word_tmp, postag_tmp, entity_tmp = '', [], []
                    word_tmp += w
                    postag_tmp.append(postags_[i])
                    entity_tmp.append(entity)
                    if entity_loc == 'E':
                        words.append(word_tmp)
                        postags.append(postag_tmp[0])
                        entities.append(entity_tmp[0])
                        word_tmp, postag_tmp, entity_tmp = '', [], []
                else:
                    words.append(w)
                    postags.append(postags_[i])
                    entities.append(entity)
            else:
                words.append(w)
                postags.append(postags_[i])
                if entity_loc == 'O':
                    entities.append('O')
                else:
                    entities.append(entity_loc + '-' + entity)

        return {'tokens':words, 'tags': postags, 'entities':entities}


class Segment_Mecab(Segment_Base):
    '''
        Mecab wrapper, Please check `Mecab <https://taku910.github.io/mecab/>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - seg_dict_path: mecab model path
    '''
    def __init__(self, cfg):
        import MeCab
        self.mecab_ins = MeCab.Tagger('-d %s ' % cfg["seg_dict_path"])
        Segment_Base.__init__(self, cfg)

    
    def seg(self, sentence, remove_stopwords=False, tags_filter=None):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True
                - tags_filter: string list, will only main special tag tokens, default is None. Some available tags are "名詞", "動詞", "助動詞", "形容詞", "助詞", "係助詞"...

            Output: dictionary with keys:
                - tokens: list of tokens
                - tags: list of detailed part-of-speech tags
        '''
        #sentence = re.sub(r"[\s\u0020-\u007f\u2000-\u206f\u3000-\u303f\uff00-\uffef]+", " ", sentence)
        sentence = re.sub(r"[^\w\d]+", " ", sentence)
        tokens, tags = [], []
        m = self.mecab_ins.parseToNode(sentence)
        while m:
            word_type = m.feature.split(',')[0]
            try:
                m.surface
            except:
                m = m.next
                continue
            if len(m.surface) < 1 :
                m = m.next
                continue
            if remove_stopwords and m.surface in self.stopwords:
                m = m.next
                continue
            if tags_filter is not None and word_type not in tags_filter:
                m = m.next
                continue
            tokens.append(m.surface)
            tags.append(word_type)
            m = m.next
        return {"tokens":tokens, "tags":tags}
    

class Segment_Keras(Segment_Base):
    '''
        Tokenizer wrote by Keras, use bi-lstm + crf, not finished, please don't use it
    '''
    def __init__(self, cfg):
        from keras.models import Model, load_model
        from ..utils import zload
        from .vocab import Vocab
        Segment_Base.__init__(self, cfg)
        self.not_cuts = re.compile('([\da-zA-Z ]+)|[\.\+\,\-\!\?\/\:\;，。！？、；：]')
        self.seg_dict = cfg['seg_dict_path']
        self.model = load_model(os.path.join(self.seg_dict, 'seg.h5'))
        self.v_word = Vocab({'cached_vocab': os.path.join(self.seg_dict, 'vocab.pkl')})
        self.zy = {'BE':0.5, 'BM':0.5, 'EB':0.5, 'ES':0.5, 'ME':0.5, 'MM':0.5, 'SB':0.5, 'SS':0.5}
        self.zy = {i:numpy.log(self.zy[i]) for i in  self.zy.keys()}
        self.max_seq_len = 32

    def viterbi(self, nodes):
        paths = {'B':nodes[0]['B'], 'S':nodes[0]['S']}
        for l in range(1,len(nodes)):
            paths_ = paths.copy()
            paths = {}
            for i in nodes[l].keys():
                nows = {}
                for j in paths_.keys():
                    if j[-1]+i in self.zy.keys():
                        nows[j+i]= paths_[j]+nodes[l][i]+self.zy[j[-1]+i]
                k = numpy.argmax(nows.values())
                paths[list(nows.keys())[k]] = list(nows.values())[k]
        return list(paths.keys())[numpy.argmax(paths.values())]
    
    def simple_cut(self, s):
        if s:
            s_id = numpy.array([self.v_word.word2id(ss) for ss in s], 'int32')
            if len(s_id) > self.max_seq_len: s_id = s_id[:self.max_seq_len]
            s_id = numpy.concatenate((s_id, numpy.zeros(self.max_seq_len-len(s_id), 'int32')))
            
            r = self.model.predict(s_id.reshape((1, len(s_id))))[0][:len(s)]
            r = numpy.log(r)
            
            nodes = [dict(zip(['B','M','E','S'], i[1:])) for i in r]
            t = self.viterbi(nodes)
            words = []
            for i in range(len(s)):
                if t[i] in ['S', 'B']:
                    words.append(s[i])
                else:
                    words[-1] += s[i]
            return words
        else:
            return []

    def seg(self, s):
        result = []
        j = 0
        for i in self.not_cuts.finditer(s):
            result.extend(self.simple_cut(s[j:i.start()]))
            result.append(s[i.start():i.end()])
            j = i.end()
        result.extend(self.simple_cut(s[j:]))
        return result

    def seg_bak(self, sentence):
        ids = [self.v_word.word2id(x) for x in sentence]
        ids2 = []
        for i in range(len(sentence)):
            ids2.append([])
            for ni in range(i-3, i+4):
                if ni < 0 or ni >= len(ids2)-1:
                    ids2[-1].append(0)
                else:
                    ids2[-1].append(ids[ni])
        ids2 = numpy.array(ids2)
        
        labels = self.model.predict(ids2)
        labels = numpy.argmax(labels, axis=1)
        print(labels)
        
        tokens = []
        token = ''
        for i in range(len(sentence)):
            token += sentence[i]
            if self.v_label.id2word[labels[i]] in ['E', 'S']:
                tokens.append(token)
                token = ''
        tokens.append(token)
        return [x for x in tokens if len(x)>0]


class Segment_Rest(Segment_Base):
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)
        self.rest_url = cfg['TOKENIZER']
    
    def seg(self, sentence, remove_stopwords = True, tags_filter = None, entities_filter = None, pos_filter = None, dep_filter=None):
        txts, tokens, entities, pos= [], [], [], []
        data = restpost(self.rest_url, {'text':sentence})
        filtereddata = {}
        for k in data: filtereddata[k] = []
        for i in range(len(data['tokens'])):
            if remove_stopwords and data['tokens'][i] in self.stopwords:
                continue
            if tags_filter is not None and 'tags' in data and data['tags'][i] not in tags_filter:
                continue
            if pos_filter is not None and 'pos' in data and data['pos'][i] not in pos_filter:
                continue
            if dep_filter is not None and 'dep' in data and data['dep'][i] not in dep_filter:
                continue
            if 'entities' in data:
                entity = data['entities'][i]
                if self.cfg['ner_name_replace'] is not None and data['entities'][i] in self.cfg['ner_name_replace']:
                    data['entities'][i] = self.cfg['ner_name_replace'][data['entities'][i]]
                if entities_filter is not None and 'entities' in data and data['entities'][i] not in entities_filter:
                    continue
            for k in data: filtereddata[k].append(data[k][i])
        return filtereddata


# natural segment
class Segment_Simple(Segment_Base):
    '''
        Tokenizer use regex

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
                    - TOKENIZER REGEX: regex string, default is [\s\.\:\;\&\'\"\/\\\(\)\[\]\{\}\%\$\#\!\?\^\&\+\`\~（）《》【】「」；：‘“’”？／。、，]
    '''
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)
        if not 'TOKENIZER_REGEX' in self.cfg:
            tokenizer_regex = ur'[\s\.\:\;\&\'\"\/\\\(\)\[\]\{\}\%\$\#\!\?\^\&\+\`\~（）《》【】「」；：‘“’”？／。、，]'
        else:
            tokenizer_regex = self.cfg['TOKENIZER_REGEX']
        self.re_punc = re.compile(tokenizer_regex)
    
    def seg(self, sentence, remove_stopwords = True):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True

            Output: dictionary with keys:
                - tokens: list of tokens
        '''
        tokens = [s.lower() for s in self.re_punc.split(sentence) if len(s)>0]
        if remove_stopwords:
            tokens = [s for s in tokens if s not in self.stopwords]
        return {'tokens': tokens}


# character level segment
class Segment_Char(Segment_Base):
    '''
        Split sentence to character list

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all keys mentioned in Segment_Base
    '''
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)

    def seg(self, sentence, remove_stopwords = True):
        ''' segment sentence to characters
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True

            Output: dictionary with keys:
                - tokens: list of characters
        '''
        tokens = []
        for s in sentence:
            s = s.strip().lower()
            if len(s) < 1: continue
            if remove_stopwords and s in self.stopwords:
                continue
            tokens.append(s)
        return {'tokens': tokens}


class Segment(object):
    '''
        Tokenizer tool, integrate with several tools 

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: 
                    - TOKENIZER: string, choose for tokenizer tool:
                        1. *corenlp*: stanford CoreNLP
                        2. *spacy*: spacy
                        3. *jieba*: jieba
                        4. *ltp*: HIT-SCIR pyltp
                        5. *mecab*: mecab
                        6. *simple*: regex
                        7. *char*: will split to char level
                        8. *http://**: will use restapi
                    - LANGUAGE: if *TOKENIZER* not exists, will check the *LANGUAGE* config:
                        1. cn: will use jieba
                        2. yue: will use jieba
                        3. en: will use spacy
                        4. jp: will use mecab

        Example Usage:
            - seg = Segment(cfg); seg.seg(sentence)
            - support __call__ method: seg(sentence)
    '''
    def __new__(cls, cfg):
        tokenizers = {'corenlp':Segment_CoreNLP, \
                      'spacy':Segment_Spacy, \
                      'jieba':Segment_Jieba, \
                      'ltp':Segment_LTP, \
                      'mecab': Segment_Mecab, \
                      'simple':Segment_Simple,\
                      'char': Segment_Char}
        languages = {'cn':'jieba', \
                     'yue':'jieba', \
                     'en':'spacy', \
                     'jp': 'mecab'}
        if 'TOKENIZER' in cfg:
            if cfg['TOKENIZER'] in tokenizers:
                return tokenizers[cfg['TOKENIZER']](cfg)
            elif 'http' in cfg['TOKENIZER']:
                return Segment_Rest(cfg) 
        if 'LANGUAGE' in cfg and cfg['LANGUAGE'] in languages:
            return tokenizers[languages[cfg['LANGUAGE']]](cfg)
        return Segment_Simple(cfg)




