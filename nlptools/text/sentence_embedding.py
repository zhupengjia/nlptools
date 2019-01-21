#!/usr/bin/env python
from pytorch_pretrained_bert.modeling import BertModel
from .tokenizer import Tokenizer_BERT

class Sentence_Embedding:
    '''
        Extract sentence embedding from bert

        Input:
            - bert_model_name: bert model file location or one of the supported model name
            - do_lower_case: bool, default is True
            - max_seq_len: int, maximum sequence length, default is 100
            - device: string, 'cpu' or 'cuda:i'
    '''
    def __init__(self, bert_model_name, do_lower_case=True, max_seq_len=100, device='cpu'):

        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = Tokenizer_BERT(bert_model_name=bert_model_name, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.max_seq_len = max_seq_len
        if torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)


    def dim(self):
        '''
            dimention of sentence embedding
        '''
        return self.encoder.config.hidden_size

    def __call__(self, sentences, batch_size=1):
        '''
            return sentence embedding
        '''
        if isinstance(sentences, str):
            sentences = [sentences]
            batch_size = 1
        for i in range(batch_size):
            starti = batch_size * i
            endi = min(batch_size * (i+1), len(sentences))
            if starti >= len(sentences):
                break
            Nsentence = endi - starti
            sentence_ids = numpy.zeros((Nsentence, self.max_seq_len), 'int')
            attention_mask = numpy.zeros((Nsentence, self.max_seq_len), 'int')
            for j in range(starti, endi):
                tokens = self.tokenizer(sentence[j])
                token_ids = numpy.concatenate(([self.vocab.CLS_ID],self.vocab.words2id(tokens),[self.vocab.SEP_ID]))
                seq_len = min(self.max_seq_len, len(token_ids)) 
                sentence_ids[j-starti, :seqlen] =  numpy.array(token_ids)[:seqlen]
                attention_mask[j-starti, :seqlen] = 1
            sentence_ids = torch.LongTensor(sentence_ids).to(self.device)
            attention_mask = torch.LongTensor(attention_mask).to(device)
            
            sequence_output, pooled_output = self.encoder(sentence_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
            yield pooled_output.cpu().numpy()


                

