import sys
sys.path.append('..')
from ailab.zoo.text_classifier.text_judgment import TextJudgment
import json
from ailab.zoo.text_classifier.data_helpers import Data_helpers
from ailab.zoo.text_classifier.model import JudgementModel
import time
import numpy as np
import pandas as pd
from ailab.text import Segment


cfg = { 
		'LANGUAGE': 'cn',
		'stopwords_path': '../data/stopwords_cn.gen',
		'w2v_word2idx': '/Users/caozx/data/word2vec/word2vec/cn/zhwiki_word2idx.pkl',
		'w2v_idx2vec': '/Users/caozx/data/word2vec/word2vec/cn/zhwiki_vectors.pkl',
		'cached_w2v': '../data/w2v_cn.pkl',
		'vec_len':100, 
		
#		'LANGUAGE':'en',
#		'w2v_word2idx': '/Users/caozx/data/word2vec/word2vec/en/word2idx_2000000.pp',
#		'w2v_idx2vec': '/Users/caozx/data/word2vec/word2vec/en/weight_2000000.npy',
#		'cached_w2v': 'data/w2v_en.pkl',
#		'vec_len':300,
		'FLAGS': 
		{'allow_soft_placement': True,
		'batch_size': 32,
		'checkpoint_every': 100,
		'dev_sample_percentage': 0.1,
		'dropout_keep_prob': 0.3,
		'embedding_dim': 101,
		'evaluate_every': 100,
		'filter_sizes': '2,3,5',
		'l2_reg_lambda': 0.3,
		'log_device_placement': False,
		'negative_data_file': '../data/no.gen',
		'num_checkpoints': 5,
		'num_epochs': 100,
		'num_filters': 128,
		'positive_data_file': '../data/yes.gen'},

		'model_path': '/Users/caozx/data/chatbot/ailab/ailab/zoo/text_classifier/test',
#		'model_file': 
#		{'out_dir': '/Users/caozx/data/chatbot/ailab/ailab/zoo/text_classifier/test/runs/1513821541',
#		 'checkpoint_dir': '/Users/caozx/data/chatbot/ailab/ailab/zoo/text_classifier/test/runs/1513821541/checkpoints'}
		} 

#data = json.dumps(cfg)
#with open('data.json', 'w') as f:
#	json.dump(data, f)

#with open('data.json', 'r') as f:
#	cfg = json.load(f)
#cfg = json.loads(cfg)

#Data_help = Data_helpers(cfg)
#Data_help.data_new(cfg['FLAGS']['positive_data_file'], cfg['FLAGS']['negative_data_file'], 2)
seg_ins = Segment(cfg)
Text_Judge = TextJudgment(cfg)
#Text_Judge.data_process(cfg['FLAGS']['positive_data_file'], cfg['FLAGS']['negative_data_file'])
#print(Text_Judge.x_train)
#batches = Data_help.batch_iter(list(zip(Text_Judge.x_train, Text_Judge.y_train)), cfg['FLAGS']['batch_size'], cfg'[FLAGS']['num_epochs'])



#Text_Judge.train()
Model = JudgementModel(cfg)


Text_Judge.predict_file(positive_file='../data/yes.gen', negative_file='../data/no.gen')

txt_file = pd.read_csv('test.txt', sep=' ')
tests = txt_file['词']; predicts=[]
for item in tests:
	Model.predict(item)
	predicts.append(Model.result)
txt_file['预测']=predicts
acc = sum(txt_file['预测']==txt_file['分类'])/len(tests)
print('test accuracy is:', acc)

query = '?'
x_text = (seg_ins.seg_sentence(query))['tokens']
x_text = [' '.join(x_text)]
#print(x_text)
#print('type of x_text is:', type(x_text))
#print('lenght of x_text is:', len(x_text))
#print("x_text[0]==' ' is:", x_text[0]=='')
t1=time.time()
Model.predict(query)
t2=time.time()
t = t2-t1
print('judgment of '+query+'is:', Model.result, 'score is:', max(Model.scores[0]))
print('time cost is:', t)

t_list =[]
for i in range(100):
	t3 = time.time()
	query = '明白了，谢谢'
	Model.predict(query)
	t4 = time.time()
	t_list.append(t4-t3)
t_list = np.array(t_list)
print('avg of cost is:', t_list.mean())
print('judgment of '+query+'is:', Model.result, 'score is:', max(Model.scores[0]))
