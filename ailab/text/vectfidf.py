#!/usr/bin/env python3
import numpy, scipy
from multiprocessing import Pool as ProcessPool

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins = None):
        self.cfg = cfg
        self.vocab = []
        for i in self.cfg['grams']
        self.vocab = vocab_ins
        


    def get_count_matrix(args, db, db_opts):
        """Form a sparse word to document count matrix (inverted index).
    
        M[i, j] = # times word i appears in document j.
        """
        # Map doc_ids to indexes
        global DOC2IDX
        db_class = retriever.get_class(db)
        with db_class(**db_opts) as doc_db:
            doc_ids = doc_db.get_doc_ids()
        DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
        # Setup worker pool
        tok_class = tokenizers.get_class(args.tokenizer)
        workers = ProcessPool(
            args.num_workers,
            initializer=init,
            initargs=(tok_class, db_class, db_opts)
        )
    
        # Compute the count matrix in steps (to keep in memory)
        logger.info('Mapping...')
        row, col, data = [], [], []
        step = max(int(len(doc_ids) / 10), 1)
        batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
        _count = partial(count, args.ngram, args.hash_size)
        for i, batch in enumerate(batches):
            logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
            for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
                row.extend(b_row)
                col.extend(b_col)
                data.extend(b_data)
        workers.close()
        workers.join()
    
        logger.info('Creating sparse matrix...')
        count_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(args.hash_size, len(doc_ids))
        )
        count_matrix.sum_duplicates()
        return count_matrix, (DOC2IDX, doc_ids)

    def load_index(self, corpus_ids=None, retrain=False):
        
        pass



