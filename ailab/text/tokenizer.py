#!/usr/bin/env python
import os, string, numpy, re, glob, json, requests
from ..utils import restpost

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Tokenizer_Base(object):
    '''
        Parent class for other tokenizer classes, please don't use this class directly

        Input:
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, stopwords_path = None, ner_name_replace = None):
        self.stopwords = {}
        self.__loadStopwords(stopwords_path)


    def __loadStopwords(self, stopwords_path):
        if stopwords_path is not None and os.path.exists(stopwords_path):
            with open(stopwords_path, encoding='utf-8') as f:
                for i in f.readlines():
                    self.stopwords[i.strip()] = ''


    def __call__(self, sentence):
        return self.seg(sentence)


class Tokenizer_CoreNLP(Tokenizer_Base):
    '''
        Stanford CoreNLP wrapper, Please check `stanford CoreNLP <https://stanfordnlp.github.io/CoreNLP/>`_ for more details

        Input:
            - corenlp_url: corenlp api url
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, corenlp_url, **args):
        Tokenizer_Base.__init__(self, **args)
        self.server_url = corenlp_url

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
                if self.ner_name_replace is not None and token['ner'] in self.ner_name_replace:
                    token['ner'] = self.ner_name_replace[token['ner']]
                if entities_filter is not None and token['ner'] not in entities_filter:
                    continue
                if len(token['lemma'])<1:
                    continue
                txts.append(token['word'])
                tokens.append(token['lemma'])
                pos.append(token['pos'])
                entities.append(token['ner'])
    
        return {"tokens":tokens, "texts":txts, "entities":entities, 'pos':pos}
   

class Tokenizer_Spacy(Tokenizer_Base):
    '''
        Spacy wrapper, Please check `Spacy <https://spacy.io/>`_ for more details

        Input:
            - spacy_model: model name or model path used for spacy.load, default is 'en'
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, spacy_model='en', **args):
        import spacy
        Tokenizer_Base.__init__(self, **args)
        self.nlp = spacy.load(spacy_model)

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
            if self.ner_name_replace is not None and entity in self.ner_name_replace:
                entitiy = self.ner_name_replace[entity]
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
   

class Tokenizer_Jieba(Tokenizer_Base):
    '''
        Jieba wrapper, Please check `Jieba <https://github.com/fxsjy/jieba>`_ for more details

        Input:
            - seg_dict_path: if the key existed, will load the user definded dict. Default is None
            - stopwords_path: the path of stopwords, default is None
    '''
    def __init__(self, seg_dict_path, **args):
        import jieba
        Tokenizer_Base.__init__(self, **args)
        if seg_dict_path is not None:
            jieba.load_userdict(seg_dict_path)
        self.nlp = jieba

    def seg(self, sentence, remove_stopwords=True):
        ''' segment sentence to words
            
            Input:
                - sentence: string
                - remove_stopwords: bool, default is True

            Output: dictionary with keys:
                - tokens: list of tokens
        '''
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
     

class Tokenizer_LTP(Tokenizer_Base):
    '''
        HIT-SCIR pyltp wrapper, Please check `pyltp <https://github.com/HIT-SCIR/pyltp>`_ for more details

        Input:
            - cws_model_path: path of cws model
            - pos_model_path: path of pos model
            - ner_model_path: path of ner model
            - parser_model_path: path of parser model, default is None
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, cws_model_path, pos_model_path, ner_model_path, parser_model_path, **args):
        Tokenizer_Base.__init__(self, **args)
        from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
        self.seg_ins = Segmentor()
        self.seg_ins.load(cws_model_path)
        self.pos_ins = Postagger()
        self.pos_ins.load(pos_model_path)
        if parser_model_path is not None and  os.path.exists(parser_model_path):
            self.parser_ins = Parser()
            self.parser_ins.load(parser_model_path)
        else:
            self.parser_ins = None
        self.ner_ins = []

        for path in sorted(glob.glob(ner_model_path)):
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

    def seg(self, sentence, remove_stopwords = True, tags_filter = None, entities_filter = None, entityjoin=False):
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
            if self.ner_name_replace is not None and entity in self.ner_name_replace:
                entity = self.ner_name_replace[entity]
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


class Tokenizer_Mecab(Tokenizer_Base):
    '''
        Mecab wrapper, Please check `Mecab <https://taku910.github.io/mecab/>`_ for more details

        Input:
            - seg_dict_path: mecab model path
            - stopwords_path: the path of stopwords, default is None
    '''
    def __init__(self, seg_dict_path, **args):
        import MeCab
        self.mecab_ins = MeCab.Tagger('-d %s ' % seg_dict_path)
        Tokenizer_Base.__init__(self, **args)

    
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
    

class Tokenizer_Rest(Tokenizer_Base):
    '''
        Tokenizer use rest_api
        
        Input:
            - tokenizer: rest_api for tokenizer
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, tokenizer, **args):
        Tokenizer_Base.__init__(self, **args)
        self.rest_url = tokenizer
    
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
                if self.ner_name_replace is not None and data['entities'][i] in self.ner_name_replace:
                    data['entities'][i] = self.ner_name_replace[data['entities'][i]]
                if entities_filter is not None and 'entities' in data and data['entities'][i] not in entities_filter:
                    continue
            for k in data: filtereddata[k].append(data[k][i])
        return filtereddata


# natural segment
class Tokenizer_Simple(Tokenizer_Base):
    '''
        Tokenizer use regex

        Input:
            - tokenizer_regex: regex string, default is [\s\.\:\;\&\'\"\/\\\(\)\[\]\{\}\%\$\#\!\?\^\&\+\`\~（）《》【】「」；：‘“’”？／。、，]
            - stopwords_path: the path of stopwords, default is None
            - ner_name_replace: dictionary, replace the entity name to the mapped name. Default is None
    '''
    def __init__(self, tokenizer_regex=None, **args):
        Tokenizer_Base.__init__(self, **args)
        if tokenizer_regex is None:
            tokenizer_regex = ur'[\s\.\:\;\&\'\"\/\\\(\)\[\]\{\}\%\$\#\!\?\^\&\+\`\~（）《》【】「」；：‘“’”？／。、，]'
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
class Tokenizer_Char(Tokenizer_Base):
    '''
        Split sentence to character list

        Input:
            - stopwords_path: the path of stopwords, default is None
    '''
    def __init__(self, **args):
        Tokenizer_Base.__init__(self, **args)

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


class Tokenizer(object):
    '''
        Tokenizer tool, integrate with several tools 

        Input:
            - tokenizer: string, choose for tokenizer tool:
                1. *corenlp*: stanford CoreNLP
                2. *spacy*: spacy
                3. *jieba*: jieba
                4. *ltp*: HIT-SCIR pyltp
                5. *mecab*: mecab
                6. *simple*: regex
                7. *char*: will split to char level
                8. *http://**: will use restapi

        Example Usage:
            - seg = Tokenizer(**args); seg.seg(sentence)
            - support __call__ method: seg(sentence)
    '''
    def __new__(cls, **args):
        tokenizers = {'corenlp':Tokenizer_CoreNLP, \
                      'spacy':Tokenizer_Spacy, \
                      'jieba':Tokenizer_Jieba, \
                      'ltp':Tokenizer_LTP, \
                      'mecab': Tokenizer_Mecab, \
                      'simple':Tokenizer_Simple,\
                      'char': Tokenizer_Char}
        if 'tokenizer' in args:
            if args['tokenizer'] in tokenizers:
                return tokenizers[args['tokenizer']](**args)
            elif 'http' in args['tokenizer']:
                return Tokenizer_Rest(**args) 
        return Tokenizer_Simple(**args)




