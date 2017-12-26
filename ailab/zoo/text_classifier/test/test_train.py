import sys
sys.path.append('..')
from ailab.zoo.text_classifier.text_judgment import TextJudgment
import json
from ailab.zoo.text_classifier.data_helpers import Data_helpers

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

Data_help = Data_helpers(cfg)
#Data_help.data_new(cfg['FLAGS']['positive_data_file'], cfg['FLAGS']['negative_data_file'], 2)

Text_Judge = TextJudgment(cfg)
#Text_Judge.data_process(cfg['FLAGS']['positive_data_file'], cfg['FLAGS']['negative_data_file'])
#print(Text_Judge.x_train)
#batches = Data_help.batch_iter(list(zip(Text_Judge.x_train, Text_Judge.y_train)), cfg['FLAGS']['batch_size'], cfg'[FLAGS']['num_epochs'])



#Text_Judge.train()

#Text_Judge.load_checkpoint()
#Text_Judge.predict(positive_file='../data/test-positive.gen', negative_file='../data/test-negative.gen')

Text_Judge.predict(positive_file='../data/yes.gen', negative_file='../data/no.gen')
#print('scores are:', Text_Judge.scores)
query = '对不起，没听清楚，不明白，不懂，不好意思'
Text_Judge.predict(query)
print('judgment of '+query+'is:', Text_Judge.result, 'score is:', max(Text_Judge.scores[0]))

query = '明白了，谢谢'
Text_Judge.predict(query)
print('judgment of '+query+'is:', Text_Judge.result, 'score is:', max(Text_Judge.scores[0]))
