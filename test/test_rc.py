#!/usr/bin/env python3
import re, json
from ailab.zoo.comprehension.DrQA.drqa import DrQA

homedir = '/Users/pengjia.zhu/'
cfg = {'APPNAME':'test', 'vec_len':300, 'w2v_word2idx':homedir+'data/word2vec/en/word2idx_2000000.pp', 'w2v_idx2vec':homedir+'data/word2vec/en/weight_2000000.npy',  'LANGUAGE':'en', 'cached_w2v':'data/w2v.pkl', 'cached_vocab':'data/vocab.pkl', 'cached_index':'data/tfidf.index', 'freqwords_path':'data/en_freqwords.txt', 'vocab_size':99055, 'drqa_reader_path':homedir+'data/read_comprehension/SQuAD/DrQA/reader/single.mdl', 'drqa_data_cache':'data/accenture_policy.pkl'}

d = DrQA(cfg)
d.load_reader()
d.vocab.save()
with open('data/accenture_policy.txt') as f:
    documents = re.split('\n',f.read())
documents = [json.loads(l)['text'] for l in documents if len(l.strip())>0]
d.train(documents)

#doc = "China emerged as one of the world's earliest civilizations in the fertile basin of the Yellow River in the North China Plain. For millennia, China's political system was based on hereditary monarchies, or dynasties, beginning with the semi-legendary Xia dynasty. Since then, China has expanded, fractured, and re-unified numerous times. In 1912, the Republic of China (ROC) replaced the last dynasty and ruled the Chinese mainland until 1949, when it was defeated by the communist People's Liberation Army in the Chinese Civil War. The Communist Party established the People's Republic of China in Beijing on 21 September 1949, while the ROC government retreated to Taiwan with its present de facto capital in Taipei. Both the ROC and PRC continue to claim to be the legitimate government of all China, though the latter has more recognition in the world and controls more territory.\n\n"

#question = 'where is ROC government'
#print(d.search_reader(question, doc))

question = 'what is the dresscode'
d.search(question)



