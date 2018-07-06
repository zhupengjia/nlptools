#!/usr/bin/env python
import sys, torch
from nlptools.zoo.modules.bucket import demo_data, prepare_lm_data
from nlptools.zoo.languagemodel.lstm_tagger import LSTMTagger

device = torch.device('cpu')

def main():

    '''
    PART I. Training
    '''
    
    inputs, targets, vocab_size, tagset_size = demo_data()
    inputs, targets = prepare_lm_data(inputs)
    print('vocab_size: {}, tagset_size: {}'.format(vocab_size, tagset_size))
    print('data before trainig:')
    print(inputs)
    print(targets)
    
    model = LSTMTagger(
            embedding_dim=8, hidden_dim=8, 
            # VERY IMPORTANT! to use the vocab_size as the tagset_size
            vocab_size=vocab_size, tagset_size=vocab_size,
            num_layers=1, device=device
    )
    
    '''
        `bucket_config = [(4, 3), (8, 2)]` means that:
        batch_size for seq_len up to 4 is 3
        batch_size for seq_len up to 8 is 2
        batch_size for seq_len more than 8 is 0
    '''
    
    model.bucket_config = [(4, 3), (8, 2)]
    model.train(inputs, targets, num_epoch=200)
   


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
