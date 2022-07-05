from typing import Dict, Tuple, List, Any
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
from utilities import utility
import csv
import sys
sys.path.append('models')
sys.path.append('Datasets')
sys.path.append('utilities')

csv.field_size_limit(sys.maxsize)
seed = 100

analyzer = utility.load_analyzer()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Rerank(object):
    def __init__(self, hyparams: Dict[str, Any]):

        print('Initializing indexes...')
        self.index_reader = utility.loader_index(hyparams['index_dir'])

        self.model = hyparams['RankModel']
        self.InferenceDataset = hyparams['InferenceDataset']
        self.dataIterate = hyparams['dataIterate']
        self.queries = hyparams['queries']

    def _init_query(self, q_id: str, rank_scores: bool = False):
        self.InferenceDataset.__init_q_docs__(q_id, self.queries[q_id])
        self.InferenceDataset.query_tokens = [q for q in analyzer.analyze(
            self.InferenceDataset.query) if q not in utility.STOP]
        print('qurey:{0}'.format(self.queries[q_id]))
        if rank_scores:
            prediction = self._rank_docs(
                self.InferenceDataset.query, self.InferenceDataset.top_docs)
            self.InferenceDataset.prediction = prediction
            # sort doc index from high to low
            self.InferenceDataset.rank = np.argsort(
                -self.InferenceDataset.prediction)

    def _rank_docs(self, query: str, docs: List[str], batch_size=64):
        inputs_data = self.dataIterate.CustomDataset(
            query, docs, self.InferenceDataset.tokenizer, device)
        inputs_iter = DataLoader(inputs_data, batch_size=batch_size, collate_fn=getattr(
            inputs_data, 'collate_fn', None))
        prediction = np.array([])
   
        with torch.no_grad():
            for i, batch in enumerate(inputs_iter):
                out = self.model(batch).detach().cpu().squeeze(-1).numpy()
                prediction = np.append(prediction, out)
        return prediction


    def _rank_doc_score(self, query: str, doc: List[str], batch_size=1):
        inputs_data = self.dataIterate.CustomDataset(
            query, doc, self.InferenceDataset.tokenizer, device)
        inputs_iter = DataLoader(inputs_data, batch_size=batch_size, collate_fn=getattr(
            inputs_data, 'collate_fn', None))

        for _, batch in enumerate(inputs_iter):
            out = self.model(batch).squeeze(-1)
        return out
