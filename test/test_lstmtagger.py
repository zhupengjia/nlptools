#!/usr/bin/env python
import sys, torch
from nlptools.text import Vocab
from nlptools.text.embedding import Embedding_Random
from nlptools.zoo.modules.bucket import demo_data, prepare_lm_data
from nlptools.zoo.languagemodel.lstm_tagger import LSTMTagger


def main():

    '''
    PART I. Training
    '''
    
    inputs, targets, word_vocab, tag_vocab = demo_data()
    inputs, targets = prepare_lm_data(inputs)
    print('vocab_size: {}, tagset_size: {}'.format(len(word_vocab), len(tag_vocab)))
    print('data before trainig:')
    print(inputs)
    print(targets)


    word_vocab.embedding = Embedding_Random(dim = 8)


    model = LSTMTagger(
            word_vocab, hidden_dim=8, 
            # VERY IMPORTANT! to use the vocab_size as the tagset_size
            tagset_size=word_vocab.vocab_size,
            num_layers=1
    )
    
    
    model.train(inputs, targets, num_epoch=200, max_words=20)
   


    sys.exit()

    '''
    PART II. TESTING
    '''
    
    model = LSTMTagger(
            embedding_dim=8, hidden_dim=8, 
            # VERY IMPORTANT! to use the vocab_size as the tagset_size
            vocab_size=vocab_size, tagset_size=vocab_size,
            num_layers=1
    )
    model.load_params('autosave.torch')
    
    inputs, targets, vocab_size, tagset_size = demo_data()
    inputs, targets = prepare_lm_data(inputs)
    
    buckets = BucketData(inputs, targets, [(4, 3), (8, 2)])
    for batch_inputs, batch_tags in buckets:
        print('batch inputs:', batch_inputs)
        xent = model.sequence_loss(batch_inputs, batch_tags)
        print('cross entropy:', xent)
        print('-' * 20)
        break
    
    
if __name__ == '__main__':
    main()
