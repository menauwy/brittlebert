import enum
from math import e
from matplotlib.lines import Line2D
from matplotlib.pyplot import twinx
from transformers import BertTokenizer
from pyserini import index
from seaborn.palettes import color_palette
from ranklist import Rerank, device, analyzer
from models.bert_model import BertRanker
from models.drmm_model import DRMMRanker
from models.dpr_model import DprRanker
from Datasets import dataIterBase, dataIterBert, dataIterDrmm, dataIterDpr, trec, trecdl
import numpy as np
from pyserini.index import IndexReader
from utilities.metrics import relative_ranking_changes
import argparse
from tqdm import tqdm
import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from copy import deepcopy
import pickle
from pytorch_lightning import seed_everything
import torch
import seaborn as sns
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter
import random


# import utilities


bert_type = 'bert-base-uncased'
dpr_question = 'facebook/dpr-question_encoder-multiset-base'
dpr_context = 'facebook/dpr-ctx_encoder-multiset-base'
project_dir = Path('/home/wang/attackrank')
bert_cache = Path('/home/wang/pretrained/bert/')
dpr_cache = Path('/home/wang/pretrained/dpr/')
glove_cache = Path('/home/wang/attackrank/Datasets/pkl/glove_vocab.pkl')
datasets = ['clueweb09', 'robust04', 'msmarco_p']

csv.field_size_limit(100000000)


def init_rerank(dataset: str = 'clueweb09', reranker_type: str = 'bert', model_fold: str = 'fold_1'):
    queries = {}
    if dataset in datasets:
        if dataset == 'msmarco_p':
            data_dir = project_dir / f"Datasets/src/{dataset}"
            documents_file = data_dir / 'top_dev.tsv'
            top_file = data_dir / 'top_dev.tsv'
            # 'queries.tsv' only inclue 200 test queries
            queries_file = data_dir / 'top_dev.tsv'
            index_dir = str(data_dir / 'indexes' / f"{dataset}_indexes")
            queries = trecdl.get_queries(queries_file)

        else:
            data_dir = project_dir / f"Datasets/src/{dataset}"
            documents_file = data_dir / 'documents.tsv'
            top_file = data_dir / 'top.tsv'
            queries_file = data_dir / 'queries.tsv'
            index_dir = str(data_dir / 'indexes' / f"{dataset}_indexes")
            queries = trec.get_queries(queries_file)

        kwargs = {'data_file': None, 'train_file': None, 'val_file': None, 'test_file': None,
                  'training_mode': None, 'rr_k': None, 'num_workers': None, 'bert_type': bert_type, 'bert_cache': bert_cache, 'freeze_bert': False}

        if reranker_type == 'bert':
            # @wang: omit f'{dataset}': model_dir = project_dir / f'trained/{dataset}/pairwise/{model_fold}/{reranker_type}/lightning_logs/version_0/checkpoints/'
            model_dir = project_dir / \
                f'trained/pairwise/{model_fold}/{reranker_type}/lightning_logs/version_0/checkpoints/'
            checkpoint = str(list(model_dir.glob('*.ckpt'))[0])
            model = BertRanker.load_from_checkpoint(
                checkpoint, **kwargs).to(device).eval()

            InferenceDataset = dataIterBert.InferenceDataset(
                documents_file, top_file, bert_type, bert_cache, DATA=dataset, device=device)
            dataIterate = dataIterBert

        elif reranker_type == 'drmm':
            model_dir = project_dir / \
                f'trained/{dataset}/pairwise/{model_fold}/{reranker_type}/lightning_logs/version_0/checkpoints/'
            checkpoint = str(list(model_dir.glob('*.ckpt'))[0])
            kwargs['vocab_file'] = glove_cache
            model = DRMMRanker.load_from_checkpoint(
                checkpoint, **kwargs).to(device).eval()
            InferenceDataset = dataIterDrmm.InferenceDataset(
                documents_file, top_file, model.vocab, DATA=dataset, device=device)
            dataIterate = dataIterDrmm

        elif reranker_type == 'dpr':
            dpr_args = {'question_model': dpr_question, 'context_model': dpr_context,
                        'dpr_cache': dpr_cache, 'loss_margin': 0.2, 'batch_size': 32}
            kwargs.update(dpr_args)
            model = DprRanker(kwargs).to(device).eval()
            InferenceDataset = dataIterDpr.InferenceDataset(
                documents_file, top_file, dpr_question, dpr_context, dpr_cache, DATA=dataset, device=device)
            dataIterate = dataIterDpr

        kwargs = {'dataset': dataset, 'RankModel': model, 'InferenceDataset': InferenceDataset,
                  'dataIterate': dataIterate, 'queries': queries, 'index_dir': index_dir}

        Reranker = Rerank(kwargs)
        return Reranker
    else:
        raise ValueError('dataset: clueweb09')


def get_rank(hyparams: Dict[str, Any]):
    """ Get ranking list for all queries."""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    predictions = []
    ranks = []
    for q_id in tqdm(hyparams['q_ids'], desc='Generating ranking list for each query:'):
        Reranker._init_query(q_id, rank_scores=True)
        """ save prediction and rank seperately
        save_dir1 = hyparams['exp_fold'] / \
            f"rank_{hyparams['Rerank_model']}" / \
            'query_rank' / f"{q_id}_prediction.json"
        save_dir2 = hyparams['exp_fold'] / \
            f"rank_{hyparams['Rerank_model']}" / \
            'query_rank' / f"{q_id}_rank.json"
        with open(save_dir1, 'w')as f:
            json.dump(Reranker.InferenceDataset.prediction.tolist(), f)
        with open(save_dir2, 'w')as f:
            json.dump(Reranker.InferenceDataset.rank.tolist(), f)
        """
        predictions.append(Reranker.InferenceDataset.prediction.tolist())
        ranks.append(Reranker.InferenceDataset.rank.tolist())
    save_dir1 = hyparams['exp_fold'] / \
        f"rank_{hyparams['Rerank_model']}" / \
        'query_rank' / f"predictions.json"
    save_dir2 = hyparams['exp_fold'] / \
        f"rank_{hyparams['Rerank_model']}" / \
        'query_rank' / f"ranks.json"
    with open(save_dir1, 'w')as f:
        json.dump(predictions, f)
    with open(save_dir2, 'w')as f:
        json.dump(ranks, f)


def get_rank_200(hyparams: Dict[str, Any]):
    """Get ranking kist for 200 queries"""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    pre_list = []
    rank_list = []
    q_id = [str(i) for i in range(1, 201)]
    for q_id in tqdm(q_id, desc='Generating ranking list for 200 query:'):
        Reranker._init_query(q_id, rank_scores=True)
        pre_list.append(Reranker.InferenceDataset.prediction.tolist())
        rank_list.append(Reranker.InferenceDataset.rank.tolist())
    save_dir1 = hyparams['exp_fold'] / \
        f"rank_{hyparams['Rerank_model']}" / '200_query'/'200_prediction.json'
    save_dir2 = hyparams['exp_fold'] / \
        f"rank_{hyparams['Rerank_model']}" / '200_query'/'200_rank.json'
    with open(save_dir1, 'w')as f:
        json.dump(pre_list, f)
    with open(save_dir2, 'w')as f:
        json.dump(rank_list, f)


def get_topdoc(hyparams: Dict[str, Any]):
    batch_ids = []
    model_type = hyparams['Rerank_model']
    file = hyparams['exp_fold'] / \
        f'rank_{model_type}' / 'query_rank' / 'ranks.json'
    with open(file, 'r')as f:
        ranks = json.load(f)
    for index, q_id in enumerate(hyparams['q_ids']):
        batch_ids.append([q_id, ranks[index][0]])

    final = hyparams['exp_fold'] / f'rank_{model_type}' / 'id_pair.json'
    with open(final, 'w')as f:
        json.dump(batch_ids, f)


def get_topdoc_200(hyparams: Dict[str, Any]):
    batch_ids = []
    model_type = hyparams['Rerank_model']
    file_path = hyparams['exp_fold'] / \
        f'rank_{model_type}' / '200_query' / '200_rank.json'
    with open(file_path, 'r')as f:
        rank = json.load(f)

    for i, q_id in enumerate(range(1, 201)):
        q_id = str(q_id)
        batch_ids.append([q_id, rank[i][0]])
    pair_path = hyparams['exp_fold'] / \
        f'rank_{model_type}' / '200_query' / 'id_pair_200.json'
    with open(pair_path, 'w')as f:
        json.dump(batch_ids, f)


def get_batch(hyparams: Dict[str, Any]):
    """get batch data(queries, docs) for specific fold"""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    working_path = project_dir / \
        f"Results/{hyparams['dataset']}/{hyparams['FOLD_NAME']}/rank_{hyparams['Rerank_model']}"
    batch_ids_path = working_path / 'id_pair.json'
    with open(batch_ids_path, 'r')as f:
        batch_ids = json.load(f)

    queries, batch_docs = [], []
    for index, q_id in enumerate(hyparams['q_ids']):
        queries.append(Reranker.queries[q_id])
        """
        _, top_docs = dataIterBase.read_top_docs(
            q_id, Reranker.InferenceDataset.documents_file, Reranker.InferenceDataset.top_file)
        """
        Reranker.InferenceDataset.__init_q_docs__(q_id, Reranker.queries[q_id])
        top_docs = Reranker.InferenceDataset.top_docs

        doc_index = batch_ids[index][1]
        batch_docs.append(top_docs[doc_index])
    out_path = working_path / 'batch_topdoc.json'
    with open(out_path, 'w')as f:
        json.dump({'queries': queries, 'batch_docs': batch_docs}, f)


def get_batch_200(hyparams: Dict[str, Any]):
    """get batch data(queries, docs) for specific fold"""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    working_path = project_dir / \
        f"Results/{hyparams['dataset']}/{hyparams['FOLD_NAME']}/rank_{hyparams['Rerank_model']}"
    batch_ids_path = working_path / '200_query' / 'id_pair_200.json'
    with open(batch_ids_path, 'r')as f:
        id_pairs = json.load(f)

    queries, batch_docs = [], []
    for index, q_id in enumerate(range(1, 201)):
        q_id = str(q_id)
        queries.append(Reranker.queries[q_id])
        _, top_docs = dataIterBase.read_top_docs(
            q_id, Reranker.InferenceDataset.documents_file, Reranker.InferenceDataset.top_file)
        doc_index = id_pairs[index][1]
        batch_docs.append(top_docs[doc_index])
    out_path = working_path / '200_query' / 'batch_topdoc_200.json'
    with open(out_path, 'w')as f:
        json.dump({'queries': queries, 'batch_docs': batch_docs}, f)


def get_batch_lastdoc(hyparams: Dict[str, Any]):
    """get batch data(queries, docs) for specific fold"""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    working_path = project_dir / \
        f"Results/{hyparams['dataset']}/{hyparams['FOLD_NAME']}/rank_{hyparams['Rerank_model']}"
    batch_ids_path = working_path / 'id_pair_last.json'
    with open(batch_ids_path, 'r')as f:
        batch_ids = json.load(f)

    queries, batch_docs = [], []
    for index, q_id in enumerate(hyparams['q_ids']):
        queries.append(Reranker.queries[q_id])
        """
        _, top_docs = dataIterBase.read_top_docs(
            q_id, Reranker.InferenceDataset.documents_file, Reranker.InferenceDataset.top_file)
        """
        Reranker.InferenceDataset.__init_q_docs__(q_id, Reranker.queries[q_id])
        top_docs = Reranker.InferenceDataset.top_docs

        doc_index = batch_ids[index][1]
        batch_docs.append(top_docs[doc_index])
    print(batch_docs[0])
    out_path = working_path / 'batch_lastdoc.json'
    with open(out_path, 'w')as f:
        json.dump({'queries': queries, 'batch_docs': batch_docs}, f)


def get_batch_lastdoc_200(hyparams: Dict[str, Any]):
    """get batch data(queries, docs) for specific fold"""
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])
    working_path = project_dir / \
        f"Results/{hyparams['dataset']}/{hyparams['FOLD_NAME']}/rank_{hyparams['Rerank_model']}/200_query"
    batch_ids_path = working_path / 'id_pair_lastdoc_200.json'
    with open(batch_ids_path, 'r')as f:
        batch_ids = json.load(f)

    queries, batch_docs = [], []
    for index, q_id in enumerate(range(1, 201)):
        q_id = str(q_id)
        queries.append(Reranker.queries[q_id])
        _, top_docs = dataIterBase.read_top_docs(
            q_id, Reranker.InferenceDataset.documents_file, Reranker.InferenceDataset.top_file)
        doc_index = batch_ids[index][1]
        batch_docs.append(top_docs[doc_index])
    out_path = working_path / 'batch_lastdoc_200.json'
    with open(out_path, 'w')as f:
        json.dump({'queries': queries, 'batch_docs': batch_docs}, f)


def get_lastdoc(hyparams: Dict[str, Any]):
    # to generate batch_ids_last ['q_id',last1doc_index]
    batch_ids_last = []
    model_type = hyparams['Rerank_model']
    file = hyparams['exp_fold'] / \
        f'rank_{model_type}' / 'query_rank' / 'ranks.json'
    with open(file, 'r')as f:
        ranks = json.load(f)
    for index, q_id in enumerate(hyparams['q_ids']):
        batch_ids_last.append([q_id, ranks[index][-1]])

    final = hyparams['exp_fold'] / f'rank_{model_type}' / 'id_pair_last.json'
    with open(final, 'w')as f:
        json.dump(batch_ids_last, f)


def get_lastdoc_200(hyparams: Dict[str, Any]):
    # to generate batch_ids_last ['q_id',last1doc_index]
    batch_ids_last = []
    model_type = hyparams['Rerank_model']
    file = hyparams['exp_fold'] / \
        f'rank_{model_type}' / '200_query' / '200_rank.json'
    with open(file, 'r')as f:
        rank = json.load(f)

    for i, q_id in enumerate(range(1, 201)):
        q_id = str(q_id)
        batch_ids_last.append([q_id, rank[i][-1]])
    pair_path = hyparams['exp_fold'] / \
        f'rank_{model_type}' / '200_query' / 'id_pair_lastdoc_200.json'
    with open(pair_path, 'w')as f:
        json.dump(batch_ids_last, f)


def get_relevance(hyparams: Dict[str, Any]):
    # get list of nature words, singular, plural
    nature = ['hurricane', 'hurricanes', 'tornadoes', 'tornado', 'earthquakes',
              'warming', 'lightning', 'prairie', 'deserts', 'precipitation', 'darkening', 'thunder']
    religion = ['hindusim', 'muslims', 'baptist', 'atheist', 'celestial',
                'quran', 'islam', 'unitarian', 'judaism', 'preach', 'mormon', 'preaching']  # 'unmarried','childless','sexually','marriages'
    month = ['january', 'february', 'march', 'april', 'may', 'june',
             'july', 'august', 'september', 'october', 'november', 'december']
    lastdoc_others = ['?', 'trombone', 'plumbing',
                      'wikipedia', 'ike', 'every', 'childbirth', 'unmarries', 'latino', 'marries', 'antennae', 'cherokee']
    topdoc = ['acceptable', '.', '##dington', 'register',
              'platform', 'resulting', 'foundation', 'hapoel', 'unacceptable', 'favour', 'comprised', 'competition', 'rayon', 'hapoel']
    # count relevance
    # docs that include this word and relevance>0
    nature_dict = {w: 0 for w in nature}
    religion_dict = {w: 0 for w in religion}
    month_dict = {w: 0 for w in month}
    lastdoc_others_dict = {w: 0 for w in lastdoc_others}
    topdoc_dict = {w: 0 for w in topdoc}
    relevance_num = 0  # count all docs who have relevance score higher than 0

    # get training data index
    file_path = project_dir / 'Datasets' / 'src' / 'clueweb09'
    with open(file_path/'folds'/'fold_1'/'train_ids.txt', 'r')as f:
        train_ids = [l.strip() for l in f]

    # check relavence
    Reranker = init_rerank(
        hyparams['dataset'], hyparams['Rerank_model'], hyparams['FOLD_NAME'])

    for q_id in tqdm(train_ids, desc='Generating top docs for 155 training query:'):
        Reranker._init_query(q_id, rank_scores=False)
        top_docs_id = Reranker.InferenceDataset.top_docs_id
        top_docs = Reranker.InferenceDataset.top_docs
        length = Reranker.InferenceDataset.length

        with open(file_path/'qrels.tsv', 'r')as f:
            for line in f:
                row = line.split()
                # for every doc in topdocs
                if row[0] == q_id and row[2] in top_docs_id:
                    doc_index = top_docs_id.index(row[2])
                    doc = top_docs[doc_index].split(" ")  # str to list
                    if int(row[3]) > 0:
                        # update relevance countings
                        relevance_num += 1
                        # check word's relevance score
                        for n in nature_dict:
                            if n in doc:
                                nature_dict[n] += 1
                        for r in religion_dict:
                            if r in doc:
                                religion_dict[r] += 1
                        for r in month_dict:
                            if r in doc:
                                month_dict[r] += 1
                        for r in lastdoc_others_dict:
                            if r in doc:
                                lastdoc_others_dict[r] += 1
                        for r in topdoc_dict:
                            if r in doc:
                                topdoc_dict[r] += 1

    # relevance number: 2568
    print('relevance number:', relevance_num)
    save_dir = project_dir / 'Results' / 'clueweb09' / 'fold_1' / 'rank_bert' / 'bias'
    with open(save_dir/'bias_words.json', 'w')as f:
        json.dump([nature_dict, religion_dict, month_dict,
                  lastdoc_others_dict, topdoc_dict], f)


project_dir = Path('/home/wang/attackrank')
data_dir = project_dir / 'Datasets' / 'src' / 'clueweb09'
documents_file = data_dir / 'documents.tsv'
top_file = data_dir / 'top.tsv'
queries_file = data_dir / 'queries.tsv'

batch_ids_path = project_dir / 'Results' / 'clueweb09' / \
    'fold_1' / 'rank_bert' / 'id_pair.json'


def get_doc_for_index():
    """for every query, the top1 doc with its trigger in the form of
        json:[{doc_id:'club...',
              contents:'doc with its best triggers'},
              {},{},{}]
        output: src/clueweb09/doc_for_index/
    """
    # get triggers
    batch_ids_trigger = _top_trigger_info()

    # get all docs
    docs = {}
    with open(documents_file, encoding='utf-8', newline='')as f:
        for doc_id, doc in csv.reader(f, delimiter='\t'):
            docs[doc_id] = doc

    # get top 40 docs with triggers
    doc_top_40 = {}
    with open(batch_ids_path, 'r')as f:
        batch_ids = json.load(f)

    for pair in batch_ids:
        q_id = pair[0]
        doc_index = pair[1]
        trigger_words = batch_ids_trigger[q_id]['trigger_words']

        top_docs_id, top_docs = [], []
        with open(top_file, 'r')as f:
            for line in f:
                row = line.split()
                if row[0] == q_id:
                    top_docs_id.append(row[2])
                    top_docs.append(docs[row[2]].lower())

        # change doc_id for triggered docs
        new_doc_id = top_docs_id[doc_index] + '_fronttrigger'
        doc = trigger_words + ' ' + top_docs[doc_index]
        doc_top_40[new_doc_id] = doc

    # turn docs and doc_top_40 in specific format
    whole_doc = []
    for k, v in docs.items():
        whole_doc.append({'id': k, 'contents': v})
    for k, v in doc_top_40.items():
        whole_doc.append({'id': k, 'contents': v})

    whole_doc_file = data_dir / 'doc_for_index' / 'whole_doc_for_index.json'
    with open(whole_doc_file, 'w')as f:
        json.dump(whole_doc, f)

# wrong version: pick out triggers without [unused*], no need any more


def _top_trigger_info():

    batch_ids_trigger = {'80': {"trigger_words": "and cork baltimore shadow royale", "trigger_token_ids": [1998, 8513, 6222, 5192, 24483], "relative_changes": 0.5454545454545454},
                         '81': {"trigger_words": "block impacted kicks conditional unless", "trigger_token_ids": [3796, 19209, 14590, 18462, 4983], "relative_changes": 0.96875},
                         '134': {"trigger_words": "registering whitley saint downhill gangs", "trigger_token_ids": [25719, 27007, 3002, 19448, 18542], "relative_changes": 0.68},
                         '109': {"trigger_words": "trigger relegation finals portland overs", "trigger_token_ids": [9495, 9591, 4399, 6734, 15849], "relative_changes": 0.5567010309278351},
                         '111': {"trigger_words": "accusations show cameroneson young", "trigger_token_ids": [13519, 2265, 7232, 21421, 2402], "relative_changes": 0.51},
                         '136': {"trigger_words": "input stab wound drewgration", "trigger_token_ids": [7953, 17079, 6357, 3881, 29397], "relative_changes": 0.21348314606741572},
                         '183': {"trigger_words": "saw proceedingswas unlike scala", "trigger_token_ids": [2387, 8931, 17311, 4406, 26743], "relative_changes": 0.78},
                         '123': {"trigger_words": "funds consisted seine political rink", "trigger_token_ids": [5029, 5031, 16470, 2576, 18416], "relative_changes": 0.8080808080808081},
                         '158': {"trigger_words": "duly legitimate thanks nantes pursuit", "trigger_token_ids": [25073, 11476, 4283, 25387, 8463], "relative_changes": 0.93},
                         '180': {"trigger_words": "holders whilst matarran sport", "trigger_token_ids": [13304, 5819, 22640, 28327, 4368], "relative_changes": 0.45454545454545453},
                         '114': {"trigger_words": "proceeds severed authorised targets kicked", "trigger_token_ids": [10951, 16574, 19256, 7889, 6476], "relative_changes": 0.35353535353535354},
                         '23': {"trigger_words": "payroll skateil acceptable ivo", "trigger_token_ids": [26854, 17260, 4014, 11701, 28346], "relative_changes": 0.94},
                         '24': {"trigger_words": "^ discharge stade retrieval transfer", "trigger_token_ids": [1034, 11889, 15649, 26384, 4651], "relative_changes": 0.7731958762886598},
                         '167': {"trigger_words": "objection chevypara resolution settled", "trigger_token_ids": [22224, 29009, 28689, 5813, 3876], "relative_changes": 0.83},
                         '94': {"trigger_words": "loan ballots kilometre vacated registered", "trigger_token_ids": [5414, 17069, 13214, 15348, 5068], "relative_changes": 0.6938775510204082},
                         '73': {"trigger_words": "serviced claylot intermediate standings", "trigger_token_ids": [22858, 5726, 10994, 7783, 11869], "relative_changes": 0.87},
                         '79': {"trigger_words": "suffix wikipedia gaulle validation rai", "trigger_token_ids": [16809, 16948, 28724, 27354, 15547], "relative_changes": 0.83},
                         '44': {"trigger_words": "} investigator osman hamish oclc", "trigger_token_ids": [1065, 14064, 28609, 28859, 12258], "relative_changes": 0.31958762886597936},
                         '145': {"trigger_words": "occupants nominally otherwise team crimean", "trigger_token_ids": [18837, 24207, 4728, 2136, 20516], "relative_changes": 0.8125},
                         '142': {"trigger_words": "and meeting critical hotspurfu", "trigger_token_ids": [1998, 3116, 4187, 25985, 11263], "relative_changes": 0.7959183673469388},
                         '77': {"trigger_words": "and electorate kraftpton rayon", "trigger_token_ids": [1998, 13694, 26680, 15857, 26810], "relative_changes": 0.9191919191919192},
                         '189': {"trigger_words": "belonged churchyard face regenerationastic", "trigger_token_ids": [6272, 19812, 2227, 20045, 20875], "relative_changes": 0.4583333333333333},
                         '97': {"trigger_words": "entirety lock satisfyinghosh storage", "trigger_token_ids": [15700, 5843, 17087, 26643, 5527], "relative_changes": 0.898989898989899},
                         '149': {"trigger_words": "justified incumbent pained dom providence", "trigger_token_ids": [15123, 7703, 22295, 14383, 11293], "relative_changes": 0.82},
                         '15': {"trigger_words": "or 501 hague institutional branch", "trigger_token_ids": [2030, 16202, 14575, 12148, 3589], "relative_changes": 0.9090909090909091},
                         '47': {"trigger_words": "reacted leningrad agreement rotheet", "trigger_token_ids": [14831, 15930, 3820, 12211, 15558], "relative_changes": 0.6989247311827957},
                         '27': {"trigger_words": "prefix } kilometres conditional hapoel", "trigger_token_ids": [17576, 1065, 3717, 18462, 24208], "relative_changes": 0.797979797979798},
                         '10': {"trigger_words": "hacked iucn extentesis meetings", "trigger_token_ids": [28719, 20333, 6698, 19009, 6295], "relative_changes": 0.18},
                         '66': {"trigger_words": "supervisory rwanda redemption savings interchange", "trigger_token_ids": [26653, 17591, 18434, 10995, 8989], "relative_changes": 0.7083333333333334},
                         '32': {"trigger_words": "behalf goldman originating banda blitz", "trigger_token_ids": [6852, 17765, 14802, 24112, 22312], "relative_changes": 0.87},
                         '179': {"trigger_words": "proceeds financially it raise needy", "trigger_token_ids": [10951, 13732, 2009, 5333, 23927], "relative_changes": 0.9484536082474226},
                         '8': {"trigger_words": "mcintyre inspiring rapvalent careful", "trigger_token_ids": [24564, 18988, 9680, 24879, 6176], "relative_changes": 0.845360824742268},
                         '129': {"trigger_words": "unless dockyard slate award scotia", "trigger_token_ids": [4983, 24874, 12796, 2400, 9676], "relative_changes": 0.5151515151515151},
                         '67': {"trigger_words": "ineligible itf branch } packs", "trigger_token_ids": [22023, 27682, 3589, 1065, 15173], "relative_changes": 0.9387755102040817},
                         '107': {"trigger_words": "entirety pavilion malabarkill between", "trigger_token_ids": [15700, 10531, 28785, 15872, 2090], "relative_changes": 0.4489795918367347},
                         '113': {"trigger_words": "filing litter sheffieldchester except", "trigger_token_ids": [15242, 19070, 8533, 25322, 3272], "relative_changes": 0.5102040816326531},
                         '133': {"trigger_words": "complain hawker hardened chiang deserved", "trigger_token_ids": [17612, 23937, 15015, 17684, 10849], "relative_changes": 0.15151515151515152},
                         '119': {"trigger_words": "##hueborn madame delle temeraire", "trigger_token_ids": [20169, 10280, 10602, 24121, 23611], "relative_changes": 0.5729166666666666},
                         '106': {"trigger_words": "mutuallymable albion solojah", "trigger_token_ids": [20271, 24088, 13392, 3948, 18878], "relative_changes": 0.44329896907216493},
                         '78': {"trigger_words": "gaulle desk exchanged stade match", "trigger_token_ids": [28724, 4624, 10573, 15649, 2674], "relative_changes": 0.9393939393939394}
                         }
    return batch_ids_trigger


def get_query_doc_score():
    """
    inputs: clueweb09/indexes_bm25
    outputs: in terminal --> MO words
    """
    index_dir = '/home/wang/attackrank/Datasets/src/clueweb09/indexes_bm25'
    index_reader = IndexReader(index_dir)

    # get triggers
    batch_ids_trigger = _top_trigger_info()

    with open(batch_ids_path, 'r')as f:
        batch_ids = json.load(f)
    # get all docs
    docs = {}
    with open(documents_file, encoding='utf-8', newline='')as f:
        for doc_id, doc in csv.reader(f, delimiter='\t'):
            docs[doc_id] = doc

    queries = trec.TREC(data_dir).get_queries()

    bm25_score = []
    for pair in batch_ids:
        bm25 = {}
        q_id = pair[0]
        doc_index = pair[1]
        trigger_words = batch_ids_trigger[q_id]['trigger_words']

        query = queries[q_id]
        top_docs_id, top_docs = [], []
        with open(top_file, 'r')as f:
            for line in f:
                row = line.split()
                if row[0] == q_id:
                    top_docs_id.append(row[2])
                    top_docs.append(docs[row[2]].lower())

        print(f'query_id: {q_id}, query:{query}')
        bm25['q_id'] = q_id
        print('top docs length: ', len(top_docs))
        bm25['top_docs_length'] = len(top_docs)
        old_score = []
        for i in range(len(top_docs_id)):
            score = index_reader.compute_query_document_score(
                top_docs_id[i], query)
            old_score.append(score)
            # print(f'{i+1:2} {top_docs_id[i]:15} {score:.5f}')

        # change orig doc to doc with triggers
        orig_doc_id = deepcopy(top_docs_id[doc_index])
        top_docs_id[doc_index] = orig_doc_id + '_fronttrigger'

        orig_doc = deepcopy(top_docs[doc_index])
        top_docs[doc_index] = trigger_words + ' ' + orig_doc

        new_score = deepcopy(old_score)
        new_score[doc_index] = index_reader.compute_query_document_score(
            top_docs_id[doc_index], query)
        # use metrics to calculate changes
        changes, relative_changes = relative_ranking_changes(
            doc_index, old_score, new_score)

        print('changes: ', changes)
        bm25['changes'] = changes
        old_rank = np.argsort(old_score[::-1]).tolist().index(doc_index)
        new_rank = np.argsort(new_score[::-1]).tolist().index(doc_index)
        print('old doc score: ',
              old_score[doc_index], ', old doc rank: ', old_rank)
        bm25['old_doc_score'] = old_score[doc_index]
        bm25['old_doc_rank'] = old_rank
        print('new doc score: ',
              new_score[doc_index], ', new doc rank: ', new_rank)
        bm25['new_doc_score'] = new_score[doc_index]
        bm25['new_doc_rank'] = new_rank
        bm25_score.append(bm25)

        top_docs_id[doc_index] = orig_doc_id
        top_docs[doc_index] = orig_doc

    out_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert' / 'bm25_score_top.json'
    with open(out_path, 'w')as f:
        json.dump(bm25_score, f)


# compare triggers for clueweb09, put pictures in folder plots
result_path = project_dir / 'Results' / 'clueweb09' / \
    'fold_1' / 'rank_bert'
plot_path = result_path / 'plots'


def _compute_similarity(A_vec, B_vec):
    dot = np.dot(A_vec, B_vec)
    normA = np.linalg.norm(A_vec)
    normB = np.linalg.norm(B_vec)
    sim = dot / (normA * normB)
    return sim


def _get_embedding_weight():
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')
    for module in rerank_model.model.bert.modules():
        if isinstance(module, torch.nn.Embedding):
            # Bert has 5 embedding layers, only add a hook to wordpiece embeddings
            print(module.weight.shape[0])
            # BertModel.embeddingsword_embeddings.weight.shape == (30522,768)
            if module.weight.shape[0] == 30522:
                return module.weight.detach().cpu().numpy()

# no use of display_scatterplot_2D


def display_scatterplot_2D(X, y, all_trigger=False, trans=False):
    random_state = 1
    n_neighbors = 3
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(
        n_components=2, random_state=random_state))
    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),)

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Make a list of the methods to be compared
    # , ("LDA", lda), ("NCA", nca)
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]
    plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.subplot(1, 3, i + 1, aspect=1)
        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        if not trans:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, label=y,
                        s=5, cmap=plt.get_cmap('hsv', len(set(y))))
        else:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, label=y,
                        s=5, cmap='Set1', alpha=0.3)
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(
                name, n_neighbors, acc_knn)
        )
    if all_trigger:
        if trans:
            fig_path = f'{result_path}/plots/all_triggers_trans_visualize.png'
        else:
            fig_path = f'{result_path}/plots/all_triggers_visualize.png'
    else:
        if trans:
            fig_path = f'{result_path}/plots/all_besttriggers_trans_visualize.png'
        else:
            fig_path = f'{result_path}/plots/all_besttriggers_visualize.png'
    plt.savefig(fig_path)


def display_pca_2D(X, y, most_frequent_ids, tokenizer, dataset, attack):
    random_state = 1
    pca = make_pipeline(StandardScaler(), PCA(
        n_components=2, random_state=random_state))
    plt.figure(figsize=(8, 6))
    pca.fit(X)
    X = pca.transform(X)
    colors = ['red', 'blue', 'yellow']
    for color, i, name in zip(colors, [0, 1, 2], ['normal words', 'normal adversarial tokens', 'best adversarial tokens']):
        plt.scatter(X[y == i, 0],
                    X[y == i, 1],
                    color=color,
                    alpha=0.3,
                    label=name)
    for id in most_frequent_ids:
        plt.text(X[id, 0]-1,
                 X[id, 1]+0.7,
                 f'{tokenizer.convert_ids_to_tokens(id)}',
                 fontsize=7)
    #plt.title('2D_PCA of all word embeddings')
    plt.xlabel('the first principal component', fontsize=14)
    plt.ylabel('the second principal component', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    if dataset == 40:
        fig_path = f'{plot_path}/{attack}/PCA_besttriggers_visualize.png'
        fig_path2 = f'{plot_path}/{attack}/PCA_besttriggers_visualize.eps'
    elif dataset == 200:
        fig_path = f'{plot_path}/{attack}_200/PCA_besttriggers_visualize.png'
    plt.savefig(fig_path)
    plt.savefig(fig_path2)


def display_tsne_2D(X, y, most_frequent_ids, tokenizer, dataset, attack):
    random_state = 1
    tsne = TSNE(n_components=2, init='pca',
                random_state=random_state, learning_rate='auto')
    plt.figure(figsize=(8, 6))

    X = tsne.fit_transform(X)
    colors = ['red', 'blue', 'yellow']
    for color, i, name in zip(colors, [0, 1, 2], ['normal words', 'normal adversarial tokens', 'best adversarial tokens']):
        plt.scatter(X[y == i, 0],
                    X[y == i, 1],
                    color=color,
                    alpha=0.3,
                    label=name)
    for id in most_frequent_ids:
        plt.text(X[id, 0]-1,
                 X[id, 1]+0.7,
                 f'{tokenizer.convert_ids_to_tokens(id)}',
                 fontsize=7)
    #plt.title('2D_tSNE of all word embeddings')
    plt.xlabel('the first dimension', fontsize=14)
    plt.ylabel('the second dimension', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    if dataset == 40:
        fig_path = f'{plot_path}/{attack}/tSNE_besttriggers_visualize.png'
        fig_path2 = f'{plot_path}/{attack}/tSNE_besttriggers_visualize.eps'
    elif dataset == 200:
        fig_path = f'{plot_path}/{attack}_200/tSNE_besttriggers_visualize.png'
    plt.savefig(fig_path2)
    plt.savefig(fig_path)

# no use of nca


def display_nca_2D(X, y):
    random_state = 1
    n_neighbors = 3
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    n_classes = len(np.unique(y))
    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),)
    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    plt.figure()
    nca.fit(X_train, y_train)
    knn.fit(nca.transform(X_train), y_train)
    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(nca.transform(X_test), y_test)
    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = nca.transform(X)
    colors = ['red', 'blue']
    for color, i, name in zip(colors, [0, 1], ['normal word', 'trigger']):
        plt.scatter(X_embedded[y == i, 0],
                    X_embedded[y == i, 1],
                    color=color,
                    alpha=0.3,
                    label=name)
    plt.title('2D_NCA of all word embeddings')
    plt.legend(loc='best')
    fig_path = f'{result_path}/plots/NCA_besttriggers_visualize.png'
    plt.savefig(fig_path)


def compare_trigger(between='norm', dataset=40, attack='topdoc'):
    """choose dataset from 40/200 query
        attack choose 'topdoc'/'lastdoc'
        between choose methods to analyze triggers"""
    # get record file
    if dataset == 40:
        record_path = result_path / f'{attack}_trigger5_record.json'
    elif dataset == 200:
        record_path = result_path / '200_query' / \
            f'{attack}_trigger5_record_200.json'
    with open(record_path, 'r')as f:
        record_list = json.load(f)
    # get embedding matrix
    embedding_weight = _get_embedding_weight()
    print('max value of all dimention:', np.amax(embedding_weight))  # 0.8728918
    print('min value of all dimention:', np.amin(
        embedding_weight))  # -0.95036876

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # compare consine_similarity, norm and visualize triggers for 6 results for one query
    if between == 'norm':
        norm_of_all_words = [np.linalg.norm(vec) for vec in embedding_weight]
        mean_norm_of_all = np.mean(norm_of_all_words)
        print('Mean Norm of all word vectors:', mean_norm_of_all)  # 1.4015461

        # get norm for 40*best triggers * 40*6 unique triggers
        best_trigger_token_ids = []
        all_trigger_token_ids = []
        for record in record_list:
            q_id = record['q_id']
            trigger_6_times = record['trigger_6_times']
            # First: get unique trigger token ids
            for i, triggers in enumerate(trigger_6_times):
                all_trigger_token_ids += triggers['trigger_token_ids']

            best_trigger = sorted(
                trigger_6_times, key=lambda t: t['relative_changes'])[-1]
            best_trigger_token_ids += best_trigger['trigger_token_ids']

        print('best shape:', len(set(best_trigger_token_ids)))
        print('all shape:', len(set(all_trigger_token_ids)))

        # calculae Norm
        best_vector = [embedding_weight[id]
                       for id in set(best_trigger_token_ids)]
        all_vector = [embedding_weight[id]
                      for id in set(all_trigger_token_ids)]

        best_norm = [np.linalg.norm(vec) for vec in best_vector]  # size 159
        all_norm = [np.linalg.norm(vec) for vec in all_vector]  # size 723

        # 1) plot hitogram of norm of all words
        fig, axes = plt.subplots(figsize=(8, 6))

        sns.kdeplot(data=norm_of_all_words, color='indianred',
                    label='norm of all words')
        plt.axvline(x=mean_norm_of_all, color='indianred', ls='--',
                    label='mean norm of all words')
        sns.kdeplot(data=best_norm, color='green',
                    label='norm of best tokens')
        plt.axvline(x=np.mean(best_norm), color='green', ls='--',
                    label='mean norm of best tokens')
        sns.kdeplot(data=all_norm, color='slateblue',
                    label='norm of all tokens')
        plt.axvline(x=np.mean(all_norm), color='slateblue', ls='--',
                    label='mean morm of all tokens')
        print(mean_norm_of_all, np.mean(best_norm), np.mean(all_norm))

        plt.xlabel('norm', fontsize=18)
        plt.ylabel('density', fontsize=18)
        plt.legend(loc='best', fontsize=12)
        #plt.title('Kde of norms of trigger words', fontsize=20)
        if dataset == 40:
            fig_path = f'{plot_path}/{attack}/kde_norm.png'
            fig_path2 = f'{plot_path}/{attack}/kde_norm.eps'
        elif dataset == 200:
            fig_path = f'{plot_path}/{attack}_200/kde_norm.png'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.savefig(fig_path2, format='eps', bbox_inches='tight')
        plt.close()

        # 2) compare average norm/cosine of trigger's KNN and random word's KNN
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import cosine_similarity

        # by defalut metric='minkowski',with p=2 equals to euclidean metric
        #neigh = NearestNeighbors(n_neighbors=100, algorithm='brute')
        neigh = NearestNeighbors(
            n_neighbors=5, algorithm='brute', metric='cosine')
        neigh.fit(embedding_weight)

        # get neighbor norm/cosine distance of all triggers
        unique_trigger_token_ids = list(
            set(all_trigger_token_ids))
        # (727,768)
        trigger_embedding = embedding_weight[unique_trigger_token_ids, :]
        neigh_dist_trigger, neigh_ind_trigger = neigh.kneighbors(
            trigger_embedding, return_distance=True)  # (727,10)
        print('neigh trigger dist:', neigh_dist_trigger)
        neigh_trigger_mean = np.mean(neigh_dist_trigger)
        print('mena dist of trigger neigh:', neigh_trigger_mean)
        # 1.0323089

        # get neighbor norm distance of as much random tokens
        random_number = len(set(all_trigger_token_ids))
        vocab_token_ids_pool = np.arange(999, 30522).tolist()
        random_token_ids_pool = [
            x for x in vocab_token_ids_pool if x not in unique_trigger_token_ids]
        random_token_ids = random.sample(
            random_token_ids_pool, random_number)
        random_embedding = embedding_weight[random_token_ids, :]

        neigh_dist_random, neigh_ind_random = neigh.kneighbors(
            random_embedding, return_distance=True)

        neigh_random_mean = np.mean(neigh_dist_random)
        print('mena dist of random neigh:', neigh_random_mean)
        # 1.0102038

        # 3)plot relative changes for all expreiments
        all_changes = []  # shape()
        for i, record in enumerate(record_list):
            trigger_6_times = record['trigger_6_times']
            all_changes.append([t['relative_changes']
                               for t in trigger_6_times])
        all_changes = np.asarray(all_changes).T

        fig, axes = plt.subplots(figsize=(16, 8))
        x = [i for i in range(len(record_list))]
        for i, line in enumerate(all_changes):
            print('average line', np.mean(line))
            if i == 0:
                plt.plot(x, line, label=f'The 0th initial:[MASK]')
            else:
                plt.plot(x, line, label=f'The {i}th random seed')
        plt.legend(loc='best', fontsize=16)
        plt.xlabel('query index', fontsize=20)
        plt.ylabel('normalized rank shift', fontsize=20)
        #plt.title('Relative ranking changes for all experiments', fontsize=20)
        if dataset == 40:
            fig_path = f'{plot_path}/{attack}/all_relative_changes.png'
            fig_path2 = f'{plot_path}/{attack}/all_relative_changes.eps'
        elif dataset == 200:
            fig_path = f'{plot_path}/{attack}_200/all_relative_changes.png'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.savefig(fig_path2, bbox_inches='tight')
        plt.close()

    if between == 'similarity':
        # first, get all best trigger ids (40,5)
        best_trigger_ids = []
        for record in record_list:
            q_id = record['q_id']
            trigger_6_times = record['trigger_6_times']

            best_trigger = sorted(
                trigger_6_times, key=lambda d: d['relative_changes'])[-1]
            best_trigger_ids.append(
                best_trigger['trigger_token_ids'])  # shape(40,5)
        exp_length = len(best_trigger_ids)  # 40
        trigger_length = len(best_trigger_ids[0])  # 5

        def _plot_triggers_similarity(i, exp_length, trigger_length, trigger_ids, q_id=None, draw=False):
            global_trigger_sim = []
            random_trigger_sim = []
            if draw:
                f, ax = plt.subplots(3, 2, figsize=(12, 18))
                ax[-1, -1].axis('off')
            vocab_most_sim = []
            trigger_most_sim = []
            for j in range(trigger_length):
                # current trigger word embedding
                cur_vector = embedding_weight[trigger_ids[i][j]]
                # 1. compare similarity between the same experiment (other 4 trigger words)
                group_vector = [embedding_weight[trigger_ids[i][r]]
                                for r in range(trigger_length) if r != j]
                group_sim = [_compute_similarity(
                    cur_vector, vec) for vec in group_vector]

                # 2. compare similarity between other query triggers(39*5)
                if q_id:  # real triggers
                    other_vector = [embedding_weight[trigger_ids[p][q]] for p in range(
                        exp_length) if p != i for q in range(trigger_length)]
                else:  # random words as triggers
                    other_vector = [embedding_weight[best_trigger_ids[p][q]] for p in range(
                        exp_length) if p != i for q in range(trigger_length)]
                other_sim = [_compute_similarity(
                    cur_vector, vec) for vec in other_vector]
                unique_other_sim = [_compute_similarity(
                    cur_vector, vec) for vec in other_vector]
                global_mean = np.mean([*group_sim, *other_sim])
                global_trigger_sim.append([*group_sim, *other_sim])

                # 3. baseline1: find most similar word with this trigger in whole vocab
                vocab_sim = [_compute_similarity(
                    cur_vector, vec) for vec in embedding_weight]
                most_sim = sorted(vocab_sim)[-2]

                # 4. baseline2: random choose 40*5-1 words, compare similarity
                np.random.seed(0)
                rand_ids = np.random.randint(
                    embedding_weight.shape[0], size=exp_length*trigger_length-1)
                random_sim = [_compute_similarity(
                    cur_vector, embedding_weight[id]) for id in rand_ids]
                random_trigger_sim.append(random_sim)

                # plot kde for similarity
                # print('global:', len(group_sim+other_sim)) #shape199
                # print('group:', group_sim) #shape 4
                # print('random:', len(random_sim))#shape 199
                # print('vocab:', len(vocab_sim))#shape 30522
                if draw:
                    c1, c2, c3, c4 = sns.color_palette('Set2', 4)[:4]
                    sns.kdeplot(data=group_sim, color=c1,
                                label='sequence similarity', ax=ax[j//2, j % 2])
                    ax[j//2, j % 2].axvline(x=np.mean(group_sim), color=c1, linestyle='--', lw=0.5,
                                            label=f'mean:{np.mean(group_sim):.4f}')
                    sns.kdeplot(data=group_sim + other_sim, color=c2,
                                label='global similarity', ax=ax[j//2, j % 2])
                    ax[j//2, j % 2].axvline(x=global_mean, color=c2, linestyle='--', lw=0.5,
                                            label=f'mean:{global_mean:.4f}')
                    sns.kdeplot(data=random_sim, color=c3, label='random similarity',
                                ax=ax[j//2, j % 2])
                    ax[j//2, j % 2].axvline(x=np.mean(random_sim), color=c3, linestyle='--', lw=0.5,
                                            label=f'mean:{np.mean(random_sim):.4f}')
                    sns.kdeplot(data=vocab_sim, color=c4, label='vocab similarity',
                                ax=ax[j//2, j % 2])
                    ax[j//2, j % 2].axvline(x=np.mean(vocab_sim), color=c4, linestyle='--', lw=0.5,
                                            label=f'mean:{np.mean(random_sim):.4f}')
                    ax[j//2, j % 2].text(most_sim-0.4, 2,
                                         f'most similar in vocab:{most_sim:.4f}', fontsize=10)
                    vocab_most_sim.append(most_sim)
                    # get the highest global similarity
                    global_sim = sorted(group_sim+other_sim, reverse=True)
                    global_most_sim = 0
                    for sim in global_sim:
                        if sim < 0.99:
                            global_most_sim = sim
                            break
                    ax[j//2, j % 2].text(global_most_sim-0.4, 1,
                                         f'most similar in global trigger:{global_most_sim:.4f}', fontsize=10)
                    trigger_most_sim.append(global_most_sim)

                    ax[j//2, j %
                        2].set_xlabel('cosine similarity', fontsize=14)
                    ax[j//2, j % 2].set_ylabel('Density', fontsize=14)
                    ax[j//2, j % 2].legend(fontsize=10)
                    if q_id:
                        ax[j//2, j % 2].set_title(
                            f'The {j+1}th adversarial token', fontsize=16)
                    else:
                        ax[j//2, j % 2].set_title(
                            f'kde for cosine similarity of random trigger word{j}', fontsize=16)
            if draw:
                if q_id:
                    fig_path = f'{plot_path}/{attack}/{q_id}_sim.png'
                    fig_path2 = f'{plot_path}/{attack}/{q_id}_sim.eps'
                else:
                    fig_path = f'{plot_path}/{attack}/random_sim.png'
                    fig_path2 = f'{plot_path}/{attack}/random_sim.eps'
                plt.savefig(fig_path, bbox_inches='tight')
                plt.savefig(fig_path2, bbox_inches='tight')
                plt.close()
            return np.array(global_trigger_sim), np.array(random_trigger_sim), vocab_most_sim, trigger_most_sim

        # for every trigger word, calculate similarity and compare with baseline
        global_trigger_similarity = np.array([])
        random_trigger_similarity = np.array([])
        vocab_most_similarity = []
        trigger_most_similarity = []
        for i in range(len(best_trigger_ids)):
            q_id = record_list[i]['q_id']
            g, r, vocab_most_sim, trigger_most_sim = _plot_triggers_similarity(
                i, exp_length, trigger_length, best_trigger_ids, q_id=q_id, draw=True)
            global_trigger_similarity = np.concatenate(
                (global_trigger_similarity, g)) if global_trigger_similarity.size else g
            random_trigger_similarity = np.concatenate(
                (random_trigger_similarity, r)) if random_trigger_similarity.size else r
            vocab_most_similarity += vocab_most_sim
            trigger_most_similarity += trigger_most_sim

        # plot most similar of vocab and trigger
        plt.figure(figsize=(10, 6))
        plt.plot(vocab_most_similarity, color='indianred',
                 label='the highest similarity among vocabulary')
        plt.plot(trigger_most_similarity, color='darkblue',
                 label='the highest similarity among tokens')
        print('trigger_most_sim:', trigger_most_similarity)
        plt.xlabel('adversary token index', fontsize=12)
        plt.ylabel('cosine similairty', fontsize=12)
        plt.legend()
        if dataset == 40:
            fig_path = f'{plot_path}/{attack}/highest_similarity.png'
        elif dataset == 200:
            fig_path = f'{plot_path}/{attack}_200/highest_similarity.png'
        plt.savefig(fig_path)
        # generate random 5 trigger words, plot kde of similairty
        np.random.seed(1)
        random_ids = np.random.randint(
            embedding_weight.shape[0], size=trigger_length)
        random_ids = np.reshape(random_ids, (1, trigger_length))
        _, _, _, _ = _plot_triggers_similarity(
            0, exp_length, trigger_length, random_ids, q_id=None, draw=True)

        # check if normal distribution of global/random trigger similarity
        # assume normal distribution when w close tp 1 and p>0.05
        global_shapiro = []  # (200,2)
        print('global_trigger_similarity:', global_trigger_similarity.shape)
        for g in global_trigger_similarity:
            s, p = stats.normaltest(g)
            global_shapiro.append(p)
        global_normal_dis = np.array(global_shapiro) > 0.05
        print('whether global similarity for enery trigger word is normal distributed?', global_normal_dis)
        # almost all false.
        global_normal_dis_mean = [
            np.mean(g) for g in global_trigger_similarity]  # (200)
        global_normal_dis_std = [np.std(g)
                                 for g in global_trigger_similarity]  # (200)

        random_shapiro = []  # (200)
        print('random_trigger_similarity:', random_trigger_similarity.shape)
        for r in random_trigger_similarity:
            s, p = stats.normaltest(r)
            random_shapiro.append(p)
        random_normal_dis = np.array(random_shapiro) > 0.05
        # print(random_shapiro)
        print('whether random similarity for enery trigger word is normal distributed?', random_normal_dis)
        random_normal_dis_mean = [
            np.mean(r) for r in random_trigger_similarity]  # (200)
        random_normal_dis_std = [np.std(r)
                                 for r in random_trigger_similarity]  # (200)
        # plot normal distribution mean and standard variance
        f, ax = plt.subplots(figsize=(12, 6))
        ax1 = ax.twinx()
        ax.plot(random_normal_dis_mean, 'orangered',
                lw=1, label='random_mean')
        ax.plot(global_normal_dis_mean, 'darkred', label='global_mean')
        ax1.plot(global_normal_dis_std, 'darkblue', label='global_std')
        ax1.plot(random_normal_dis_std, 'royalblue',
                 lw=1, label='random_std')
        ax.set_xlabel('adversarial token index', fontsize=16)
        ax.set_ylabel('mean value', color='darkred', fontsize=16)
        ax.set_ylim([0, 0.55])
        ax1.set_ylabel('standard deviation', color='darkblue', fontsize=16)
        ax1.set_ylim([0.03, 0.22])
        ax.legend(loc=1, fontsize=12)
        ax1.legend(loc=2, fontsize=12)
        # plt.title(
        #    'comparision of distribution parameters of different triggers', fontsize=14)
        if dataset == 40:
            fig_path = f'{plot_path}/{attack}/Distribution_parameters.png'
            fig_path2 = f'{plot_path}/{attack}/Distribution_parameters.eps'
        elif dataset == 200:
            fig_path = f'{plot_path}/{attack}_200/Distribution_parameters.png'
        plt.savefig(fig_path)
        plt.savefig(fig_path2)
        plt.close()

    # compare relative_changes, triggers for 40 queries and visulaize them using pca&nca
    if between == 'viz':
        # get besttriggers for every query:
        changes = []
        trigger_vectors = np.array([])
        trigger_labels = np.zeros((embedding_weight.shape[0],))
        trigger_labels_trans = np.zeros((embedding_weight.shape[0],))
        trigger_labels_all = np.zeros((embedding_weight.shape[0],))
        trigger_labels_all_trans = np.zeros((embedding_weight.shape[0],))

        all_best_unique_id = []
        for i, record in enumerate(record_list):
            q_id = record['q_id']
            trigger_6_times = record['trigger_6_times']
            trigger_token_ids_all = [
                id for t in trigger_6_times for id in t['trigger_token_ids']]

            best_trigger = sorted(
                trigger_6_times, key=lambda d: d['relative_changes'])[-1]
            triggers_token_ids = best_trigger['trigger_token_ids']
            all_best_unique_id += triggers_token_ids

            # shape(40,)
            changes.append(best_trigger['relative_changes'])
            vector = np.array([embedding_weight[id]
                               for id in triggers_token_ids])
            # trigger_vectors:shape(200,768)
            trigger_vectors = np.concatenate(
                (trigger_vectors, vector), axis=0) if trigger_vectors.size else vector
            # trigger_labels: shape(200,)
            trigger_labels[triggers_token_ids] = i+1
            trigger_labels_trans[triggers_token_ids] = 1
            trigger_labels_all[trigger_token_ids_all] = i+1
            trigger_labels_all_trans[trigger_token_ids_all] = 1
            trigger_labels_all_trans[triggers_token_ids] = 2

        # First: get norm of 5triggers for every query
        norm = []
        for i in range(len(record_list)):
            vector = trigger_vectors[5*i:5*i+5, :]
            norm_vec = np.mean([np.linalg.norm(vec) for vec in vector])
            norm.append(norm_vec)

        fig, axes = plt.subplots(figsize=(12, 8))
        x = [i for i in range(len(record_list))]
        axes.plot(x, norm, color='red', marker='o')
        axes.set_xlabel('query number', fontsize=14)
        axes.set_ylabel('mean norm of triggers', color='red', fontsize=14)

        axes2 = axes.twinx()
        axes2.plot(x, changes, color='blue', marker='o')
        axes2.set_ylabel(
            'relative changes on rankings with triggers', color='blue', fontsize=14)
        plt.title(
            f'Norm of Best triggers for every query (mean norm:{np.mean(norm)})', fontsize=20)
        if dataset == 40:
            fig_path = f'{plot_path}/{attack}/all_besttrigger_norm.png'
        elif dataset == 200:
            fig_path = f'{plot_path}/{attack}_200/all_besttrigger_norm.png'
        plt.savefig(fig_path)

        # Second: visualize trigger vectors with annotations of the most frequent unique ids
        print('start')
        all_best_id_count = Counter(
            all_best_unique_id).most_common()  # decending order
        most_frequent_id = [
            pair[0] for pair in all_best_id_count if pair[1] > 1]  # list[id]

        display_pca_2D(embedding_weight,
                       trigger_labels_all_trans, most_frequent_id, tokenizer, dataset, attack)
        display_tsne_2D(embedding_weight,
                        trigger_labels_all_trans, most_frequent_id, tokenizer, dataset, attack)
        # (embedding_weight, trigger_labels,all_trigger = False, trans = False)
        # display_scatterplot_2D(
        #    embedding_weight, trigger_labels_trans, all_trigger=False, trans=True)
        # print('finish1')
        # display_scatterplot_2D(embedding_weight, trigger_labels_all, all_trigger = True, trans = Flase)
        # display_scatterplot_2D(
        #    embedding_weight, trigger_labels_all_trans, all_trigger=True, trans=True)
        # print('finish2')
        # Third: cosine similarity??

        # Fourth: all relative changes shown in scatter plot


def pca_bias_tokens():
    embedding_weight = _get_embedding_weight()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # pca label: 0: clueweb09, 1: TREC-DL, 2: both
    # nature related tokens
    nature_tokens = ["hurricane", "hurricanes", "tornadoes", "tornado", "earthquakes",
                     "warming", "lightning", "prairie", "deserts", "precipitation", "darkening", "thunder"]
    nature_label = np.array([2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0])
    nature_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in nature_tokens]
    nature_vectors = np.concatenate([embedding_weight[id]
                                     for id in nature_token_ids])
    nature_length = len(nature_tokens)

    # religion related tokens
    religion_tokens = ["hinduism", "muslims", "baptist", 'unitarian', 'judaism',
                       "atheist", "celestial", "quran", "islam", "preach", "mormon", "preaching", 'catholicism', 'preached', 'christianity', 'jesus', 'archangel', 'evangelist', 'buddhism', 'sermons', 'psalms']
    religion_label = np.array(
        [2, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1])
    religion_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in religion_tokens]
    religion_vectors = np.concatenate([embedding_weight[id]
                                       for id in religion_token_ids])
    religion_length = len(religion_tokens)

    # ethnic tokens
    ethnic_tokens = ['viking', 'cherokee',
                     'latino', 'indians', 'seminole', 'negro', 'mexican', 'latin', 'italianate', 'arabic', 'haitian']
    ethnic_label = np.array([0, 2, 2, 0, 0, 0, 1, 1, 2, 1, 1])
    ethnic_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in ethnic_tokens]
    ethnic_vectors = np.concatenate([embedding_weight[id]
                                     for id in ethnic_token_ids])
    ethnic_length = len(ethnic_tokens)

    # medical tokens
    medical_tokens = ['antibiotics', 'ethanol', 'testosterone', 'cardiac',
                      'biotechnology', 'alexia', 'bacterial', 'diabetes',
                      'obesity', 'infection', 'malaria', 'dentistry',
                      'vitamin', 'chloride', 'surgeons', 'pregnancy',
                      'cholera', 'arthritis', 'nurses', 'hormones',
                      'neuroscience', 'diseases', 'autism', 'surgical',
                      'influenza', 'tuberculosis', 'dentist', 'tumors',
                      'inflammation']
    medical_label = np.array([2, 2, 2, 0,
                              2, 2, 2, 2,
                              0, 0, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1])
    medical_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in medical_tokens]

    medical_vectors = np.concatenate(
        [embedding_weight[id] for id in medical_token_ids])
    medical_length = len(medical_tokens)

    # get pca in 2D:
    random_state = 1
    pca = make_pipeline(StandardScaler(), PCA(
        n_components=2, random_state=random_state))
    X = np.concatenate((nature_vectors, religion_vectors,
                       ethnic_vectors, medical_vectors), axis=0)

    print(nature_vectors.shape,  religion_vectors.shape,
          ethnic_vectors.shape, medical_vectors.shape)
    pca.fit(X)
    X_pca = pca.transform(X)

    print(nature_length, religion_length,
          ethnic_length, medical_length, len(X_pca))

    # plot 4 categories
    plt.figure(figsize=(8, 6))
    #colors = ['red', 'blue', 'green']
    colors = ['#F1D77E', '#B1CE46', '#5F97D2']  # 82B0D2
    for color, i, label in zip(colors, [0, 1, 2], ['ClueWeb09', 'TREC-DL', 'both']):
        X_nature = X_pca[:nature_length]
        plt.scatter(X_nature[nature_label == i, 0],
                    X_nature[nature_label == i, 1],
                    color=color,
                    marker='o'
                    )  # alpha=0.5
        X_religion = X_pca[nature_length:nature_length+religion_length]
        plt.scatter(X_religion[religion_label == i, 0],
                    X_religion[religion_label == i, 1],
                    color=color,
                    marker='<')
        X_ethnic = X_pca[nature_length +
                         religion_length:nature_length+religion_length+ethnic_length]
        plt.scatter(X_ethnic[ethnic_label == i, 0],
                    X_ethnic[ethnic_label == i, 1],
                    color=color,
                    marker='*')
        X_medical = X_pca[nature_length+religion_length+ethnic_length:]
        plt.scatter(X_medical[medical_label == i, 0],
                    X_medical[medical_label == i, 1],
                    color=color,
                    marker='D')
        # legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#F1D77E', label='ClueWeb09'),
                           Patch(facecolor='#B1CE46',
                                 label='TREC-DL'),
                           Patch(facecolor='#5F97D2', label='both'),
                           Line2D([0], [0], color='w', markerfacecolor='black', markersize=10,
                                  marker='o', label='nature-related'),
                           Line2D([0], [0], color='w', marker='<', markerfacecolor='black', markersize=10,
                                  label='religion-related'),
                           Line2D([0], [0], color='w', markerfacecolor='black', markersize=15,
                                  marker='*', label='ethnicity-related'),
                           Line2D([0], [0], color='w', marker='D', markerfacecolor='black', markersize=8, label='medicine-related'), ]
        plt.legend(handles=legend_elements, fontsize=12)
        plt.xlabel('the first principal component', fontsize=16)
        plt.ylabel('the second principal component', fontsize=16)
        plot_path = Path('/home/wang/attackrank/Results/paper_plot')
        plt.savefig(plot_path/'pca_bias.eps',
                    format='eps', bbox_inches='tight')
        plt.savefig(plot_path/'pca_bias.png', bbox_inches='tight')


def pca_bias_tokens_inplace():
    embedding_weight = _get_embedding_weight()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # pca label: 0: clueweb09, 1: TREC-DL, 2: both
    # nature related tokens
    nature_tokens = ["hurricane", "hurricanes", "tornadoes", "tornado", "earthquakes",
                     "warming", "lightning", "deserts", "precipitation",
                     "darkening", "thunder",
                     'meteorological', 'typhoon', 'storms', 'noah', 'exodus',
                     'drought', 'sunrise', 'rainfall']
    nature_label = np.array([2, 2, 2, 2, 2,
                            1, 0, 1, 2,
                             2, 2,
                             2, 1, 1, 1, 1,
                             1, 0, 0])
    nature_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in nature_tokens]
    nature_vectors = np.concatenate([embedding_weight[id]
                                     for id in nature_token_ids])
    nature_length = len(nature_tokens)

    # religion related tokens
    religion_tokens = ["hinduism", "muslims", "baptist", 'judaism',
                       "quran", "islam", "preach",
                       "mormon", "preaching", 'christianity',
                       'sermons', 'psalms',
                       'biblical', 'bible', 'satan', 'archangel', 'bless',
                       'gospel', 'synagogue', 'religious', 'parish']
    religion_label = np.array([
        2, 2, 2, 1,
        2, 0, 0,
        1, 1, 1,
        2, 1,
        1, 1, 1, 2, 1,
        0, 0, 0, 0])
    religion_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in religion_tokens]
    religion_vectors = np.concatenate([embedding_weight[id]
                                       for id in religion_token_ids])
    religion_length = len(religion_tokens)

    # ethnic tokens
    ethnic_tokens = ['viking', 'cherokee', 'latino', 'seminole',
                     'negro', 'mexican', 'latin', 'haitian',
                     'hispanic', 'texans', 'romans', 'hellenistic', 'americana',
                     'hawaiian', 'coptic', 'greek', 'ghanaian', 'afrikaans',
                     'nigerian', 'egyptian']
    ethnic_label = np.array([1, 2, 2, 2,
                             2, 1, 2, 0,
                             2, 2, 1, 1, 1,
                             2, 2, 1, 0, 0,
                             0, 0])
    ethnic_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in ethnic_tokens]
    ethnic_vectors = np.concatenate([embedding_weight[id]
                                     for id in ethnic_token_ids])
    ethnic_length = len(ethnic_tokens)

    # medical tokens
    medical_tokens = ['antibiotics', 'testosterone', 'biotechnology',
                      'alexia', 'infection',
                      'malaria', 'vitamin', 'surgeons',
                      'pregnancy', 'arthritis', 'hormones',
                      'diseases', 'autism', 'surgical', 'influenza',
                      'tuberculosis', 'inflammation',
                      'congenital', 'abortion', 'thyroid', 'symptoms',
                      'anatomical', 'therapist', 'surgery', 'insulin', 'medical',
                      'blood', 'infirmary', 'prescription', 'protein', 'flu',
                      'medicare', 'med', 'murals', 'syndrome', 'pneumonia',
                      'epidemic']
    medical_label = np.array([2, 1, 1,
                             2, 2,
                             1, 1, 2,
                             1, 1, 1,
                             1, 1, 1, 2,
                             2, 1,
                             1, 2, 1, 1,
                             1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1,
                             1, 1, 0, 0, 0,
                             0])
    medical_token_ids = [tokenizer.encode(
        token, add_special_tokens=False) for token in medical_tokens]

    medical_vectors = np.concatenate(
        [embedding_weight[id] for id in medical_token_ids])
    medical_length = len(medical_tokens)

    # get pca in 2D:
    random_state = 1
    pca = make_pipeline(StandardScaler(), PCA(
        n_components=2, random_state=random_state))
    X = np.concatenate((nature_vectors, religion_vectors,
                       ethnic_vectors, medical_vectors), axis=0)
    print(nature_vectors.shape,  religion_vectors.shape,
          ethnic_vectors.shape, medical_vectors.shape)
    pca.fit(X)
    X_pca = pca.transform(X)

    print(nature_length, religion_length,
          ethnic_length, medical_length, len(X_pca))
    # plot 4 categories
    plt.figure(figsize=(8, 6))
    #colors = ['red', 'blue', 'green']
    colors = ['#F1D77E', '#B1CE46', '#5F97D2']  # 82B0D2
    for color, i, label in zip(colors, [0, 1, 2], ['ClueWeb09', 'TREC-DL', 'both']):
        X_nature = X_pca[:nature_length]
        plt.scatter(X_nature[nature_label == i, 0],
                    X_nature[nature_label == i, 1],
                    color=color,
                    marker='o'
                    )  # alpha=0.5
        X_religion = X_pca[nature_length:nature_length+religion_length]
        plt.scatter(X_religion[religion_label == i, 0],
                    X_religion[religion_label == i, 1],
                    color=color,
                    marker='<')
        X_ethnic = X_pca[nature_length +
                         religion_length:nature_length+religion_length+ethnic_length]
        plt.scatter(X_ethnic[ethnic_label == i, 0],
                    X_ethnic[ethnic_label == i, 1],
                    color=color,
                    marker='*')
        X_medical = X_pca[nature_length+religion_length+ethnic_length:]
        plt.scatter(X_medical[medical_label == i, 0],
                    X_medical[medical_label == i, 1],
                    color=color,
                    marker='D')
        # legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#F1D77E', label='ClueWeb09'),
                           Patch(facecolor='#B1CE46',
                                 label='TREC-DL'),
                           Patch(facecolor='#5F97D2', label='both'),
                           Line2D([0], [0], color='w', markerfacecolor='black', markersize=10,
                                  marker='o', label='nature-related'),
                           Line2D([0], [0], color='w', marker='<', markerfacecolor='black', markersize=10,
                                  label='religion-related'),
                           Line2D([0], [0], color='w', markerfacecolor='black', markersize=15,
                                  marker='*', label='ethnicity-related'),
                           Line2D([0], [0], color='w', marker='D', markerfacecolor='black', markersize=8, label='medicine-related'), ]
        plt.legend(handles=legend_elements, fontsize=12)
        plt.xlabel('the first principal component', fontsize=16)
        plt.ylabel('the second principal component', fontsize=16)
        plot_path = Path('/home/wang/attackrank/Results/paper_plot')
        plt.savefig(plot_path/'pca_bias_inplace.eps',
                    format='eps', bbox_inches='tight')
        plt.savefig(plot_path/'pca_bias_inplace.png', bbox_inches='tight')


if __name__ == '__main__':

    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('FOLD_NAME', type=str,
                    help='Name of the fold. e.g. fold_1')
    ap.add_argument('Rerank_model', type=str, default='bert',
                    help='Name of the rerank model, e.g. bert/drmm')
    ap.add_argument('--dataset', type=str, default='clueweb09',
                    help='Dataset:clueweb09/msmarco_p')
    ap.add_argument('--top_k_docs', type=int,
                    help='Consider top k docs in the rankinglist')  # necessary?
    ap.add_argument('--mode', type=str, help='rankinglist, or ...')
    args = ap.parse_args()

    # get q_id
    q_ids_dir = project_dir / 'Datasets/src' / args.dataset
    exp_fold = project_dir / 'Results' / args.dataset / args.FOLD_NAME
    if args.dataset == 'msmarco_p':
        q_id_file = q_ids_dir / 'queries_dev_978.txt'
    else:
        q_id_file = q_ids_dir / 'folds' / args.FOLD_NAME / 'test_ids.txt'
    with open(q_id_file, 'r')as f:
        q_ids = [l.strip() for l in f]

    args.q_ids = q_ids
    args.exp_fold = exp_fold

    random_seed = 123
    seed_everything(random_seed, workers=True)

    if args.mode == 'rank':
        get_rank(vars(args))
    elif args.mode == 'rank_200':
        get_rank_200(vars(args))
    elif args.mode == 'topdoc':
        get_topdoc(vars(args))
    elif args.mode == 'topdoc_200':
        get_topdoc_200(vars(args))
    elif args.mode == 'batch':
        get_batch(vars(args))
    elif args.mode == 'batch_200':
        get_batch_200(vars(args))
    elif args.mode == 'batch_lastdoc':
        get_batch_lastdoc(vars(args))
    elif args.mode == 'batch_lastdoc_200':
        get_batch_lastdoc_200(vars(args))
    elif args.mode == 'lastdoc':
        get_lastdoc(vars(args))
    elif args.mode == 'lastdoc_200':
        get_lastdoc_200(vars(args))
    elif args.mode == 'bias':
        get_relevance(vars(args))
    elif args.mode == 'indexdoc':
        get_doc_for_index()
    elif args.mode == 'bm25score':
        get_query_doc_score()
    elif args.mode == 'compare_trigger':
        # dataset='200'/'40'
        # attack='topdoc'/'lastdoc'
        # 'norm' contain part of knn with norm and cosine as distance
        # compare_trigger(between='norm', dataset=40,
        #                attack='topdoc')
        #compare_trigger(between='norm', dataset=40, attack='topdoc')
        #compare_trigger(between='similarity', dataset=40, attack='topdoc')
        compare_trigger(between='viz', dataset=40, attack='topdoc')
        #compare_trigger(between='viz', dataset=200, attack='lastdoc')
    elif args.mode == 'pca_bias':
        pca_bias_tokens()
    elif args.mode == 'pca_bias_inplace':
        pca_bias_tokens_inplace()
    else:
        raise ValueError(f"no such mode:{args.mode}")
