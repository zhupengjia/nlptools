#!/usr/bin/env python
import os, string, numpy, re, glob

class Segment_Base(object):
    def __init__(self, cfg, stopwords=True):
        self.stopwords = {}
        if stopwords and 'stopwords_path' in cfg:
            self.__loadStopwords(cfg['stopwords_path'])
    
    def __loadStopwords(self, stopwords_path):
        if stopwords_path is not None and os.path.exists(stopwords_path):
            with open(stopwords_path) as f:
                for i in f.readlines():
                    self.stopwords[i.strip()] = ''

class Segment_Spacy(Segment_Base):
    def __init__(self, cfg):
        import spacy
        Segment_Base.__init__(self, cfg)
        self.nlp = spacy.load(cfg['LANGUAGE'])
    
    def seg(self, sentence, remove_stopwords = False, tags_filter = None, entities_filter = None, pos_filter = None, dep_filter=None):
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
            if entities_filter is not None and token.ent_type_ not in entities_filter:
                continue
            if len(token.lemma_)<1:
                continue
            txt = token.text.strip()
            if len(txt) < 1 : continue
            txts.append(txt)
            tokens.append(token.lemma_)
            tags.append(token.tag_)
            entities.append(token.ent_type_)
            pos.append(token.pos_)
            dep.append(token.dep_)
            
        return {"tokens":tokens, "tags":tags, "texts":txts, "entities":entities, 'pos':pos, 'dep':dep}
   
    def seg_sentence(self, sentence, remove_stopwords=True, pos_filter=['PROPN','NOUN','ADJ','PRON','ADV']):
        return self.seg(sentence, remove_stopwords=remove_stopwords, pos_filter=pos_filter)


class Segment_Jieba(Segment_Base):
    def __init__(self, cfg):
        import jieba
        Segment_Base.__init__(self, cfg)
        if 'seg_dict_path' in cfg:
            jieba.load_userdict(cfg['seg_dict_path'])
        self.nlp = jieba

    def seg(self, sentence, remove_stopwords=False):
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
     
    def seg_sentence(self, sentence):
        return self.seg(sentence, remove_stopwords=True)


class Segment_LTP(Segment_Base):
    def __init__(self, cfg):
        Segment_Base.__init__(self, cfg)
        from pyltp import Segmentor, Postagger, NamedEntityRecognizer
        self.seg_ins = Segmentor()
        self.seg_ins.load(cfg['cws_model_path'])
        self.pos_ins = Postagger()
        self.pos_ins.load(cfg['pos_model_path'])
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

    def seg(self, sentence, tags_filter = None, entities_filter = None, entityjoin=True):
        words_ = self.seg_ins.segment(sentence)
        postags_ = self.pos_ins.postag(words_)
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
            entity = re.split('-', entities_[i])
            if len(entity) > 1:
                entity_loc, entity = entity[0], entity[1]
            else:
                entity_loc, entity = 'O', 'O'
            if entity in self.cfg['ner_name_replace']:
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
    def __init__(self, cfg):
        import MeCab
        self.mecab_ins = MeCab.Tagger('-d %s ' % cfg["seg_dict_path"])
        Segment_Base.__init__(self, cfg)


    def seg(self, sentence, remove_stopwords=False, tags_filter=None):
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
    
    def seg_sentence(self, sentence):
        tags_filter = ["名詞", "動詞", "助動詞", "形容詞"]
        return self.seg(sentence, remove_stopwords=True, tags_filter=tags_filter)

    def seg_smart(self, sentence):
        tokens = self.seg_sentence(sentence)['tokens']

class Segment_Keras(Segment_Base):
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

class Segment(object):
    def __new__(cls, cfg):
        if cfg['LANGUAGE'] == 'en':
            return Segment_Spacy(cfg)
        elif cfg['LANGUAGE'] in ['yue', 'cn']:
            return Segment_Jieba(cfg)
        elif cfg['LANGUAGE'] == 'jp':
            return Segment_Mecab(cfg)
        #elif cfg['LANGUAGE'] in ['yue']:
        #    return Segment_Keras(cfg)
        else:
            raise('Error! %s language is not supported'%cfg['LANGUAGE'])




