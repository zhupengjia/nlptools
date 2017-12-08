import sys
sys.path.append('..')
from text_judgment import TextJudgment
import json


#cfg = { 
#		'LANGUAGE':'cn',
#		'w2v_word2idx': '/Users/caozx/data/word2vec/word2vec/cn/zhwiki_word2idx.pkl',
#		'w2v_idx2vec': '/Users/caozx/data/word2vec/word2vec/cn/zhwiki_vectors.pkl',
#		'cached_w2v': '../data/w2v_cn.pkl',
#		'vec_len':100, 
		
#		'LANGUAGE':'en',
#		'w2v_word2idx': '/Users/caozx/data/word2vec/word2vec/en/word2idx_2000000.pp',
#		'w2v_idx2vec': '/Users/caozx/data/word2vec/word2vec/en/weight_2000000.npy',
#		'cached_w2v': 'data/w2v_en.pkl',
#		'vec_len':300, 
#		'FLAGS':
#		{"dev_sample_percentage": .1,
#		 "positive_data_file": "../data/positive.gen",
#		 "negative_data_file": "../data/negative.gen",
#		 "embedding_dim":101,
#		 "filter_sizes": "3,4,5",
#		 "num_filters": 128,
#		 "dropout_keep_prob": 0.5,
#		 "l2_reg_lambda": 0.0003,
#		 "batch_size": 32,
#		 "num_epochs": 50,
#		 "evaluate_every": 100,
#		 "checkpoint_every": 100,
#		 "num_checkpoints": 5, 
#		 "allow_soft_placement": True,
#		 "log_device_placement": False
#		}}

#data = json.dumps(cfg)
#with open('data.json', 'w') as f:
#	json.dump(data, f)

with open('data.json', 'r') as f:
	cfg = json.load(f)
cfg = json.loads(cfg)


Text_Judge = TextJudgment(cfg)
#Text_Judge.train()
Text_Judge.load_checkpoint()


Text_Judge.predict(positive_file=cfg['FLAGS']['positive_data_file'], negative_file=cfg['FLAGS']['negative_data_file'])
query = '明白了'
Text_Judge.predict(query)
print('judgment of '+query+'is:', Text_Judge.result)

query = '不了解'
result = Text_Judge.predict(query)
print('judgment of '+query+'is:', Text_Judge.result)
