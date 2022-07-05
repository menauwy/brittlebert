"""
This file measures attack effectiveness in both global and local manner.

For each query in the dataset, we split the pre-retrieved document list 
into two subsets according to the ranking order. The subset with higher 
relevant scores is sampled to run demotion task, while the other subset 
is used to run promotion task. Both tasks will run multiple times with 
different random seeds.

Figure 2 in the short paper can be reproducted by averaging results of all 
the queries in the dataset.
"""

from ranklist import device
import brittle.attack_methods as attack_methods
from copy import deepcopy
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from brittle.attack_run import init_rerank  # get reranker model
from transformers import BertTokenizer  # get related tokenizer
from pytorch_lightning import seed_everything
import matplotlib.pylab as plt
import glob
import seaborn as sns
import pandas as pd
from utilities.utility import get_embedding_weight, extracted_grads, add_hooks, get_ranking_scores, get_new_score, get_average_grad_for_docstart, get_best_candidates_topdoc_docstart, get_best_candidates_lastdoc_docstart

def get_accuracy(doc_index, orig_predictions, new_predictions, direction):
    """
    Compute relative ranking shift
    """
    doc_num = len(orig_predictions)
    orig_ranks = np.argsort(-orig_predictions)
    orig_position = list(orig_ranks).index(doc_index)

    new_ranks = np.argsort(-new_predictions)
    new_position = list(new_ranks).index(doc_index)

    try:
        if direction == 'demotion':
            relative_changes = (new_position-orig_position) / \
                (doc_num-1-orig_position)
        if direction == 'promotion':
            relative_changes = (orig_position-new_position) / orig_position
    except:
        relative_changes = 0.0
    return relative_changes


def run_model(q_id_index, q_id, rerank_model, direction='demotion', dataset='msmarco_p'):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    if dataset == 'clueweb09':
        trigger_token_length = 5
    if dataset == 'msmarco_p':
        trigger_token_length = 3
    num_candidates = 40
    seed_everything(random_seed, workers=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
    add_hooks(rerank_model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        rerank_model)  

    project_dir = Path(__file__).parent.absolute()
    if dataset == 'clueweb09':
        rank_path = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query'
        with open(rank_path/'200_rank.json', 'r')as f:
            ranks = json.load(f)
        with open(rank_path/'200_prediction.json', 'r')as f:
            predictions = json.load(f)
    if dataset == 'msmarco_p':
        rank_path = project_dir / 'Results/msmarco_p/fold_1/rank_bert/query_rank'
        with open(rank_path/'ranks.json', 'r')as f:
            ranks = json.load(f)
        with open(rank_path/'predictions.json', 'r')as f:
            predictions = json.load(f)

    query_record = {'q_id': q_id, 'query': rerank_model.queries[q_id]}
    # 5 random seeds
    samples = []
    for x in range(5):
        query = rerank_model.queries[q_id]
        rerank_model.InferenceDataset.__init_q_docs__(
            q_id, rerank_model.queries[q_id])
        top_docs = rerank_model.InferenceDataset.top_docs

        if direction == 'demotion':
            doc_ranks = np.random.randint(low=0, high=50, size=10)
        if direction == 'promotion':
            doc_ranks = np.random.randint(
                low=50, high=min(100, len(top_docs)), size=10)

        doc_indexes = [ranks[q_id_index][r]
                       for r in doc_ranks]  # 10 random docs
        sample_docs = [top_docs[index] for index in doc_indexes]

        query_token_id = tokenizer.encode(query, add_special_tokens=False)
        trigger_start_position = len(query_token_id)+2

        orig_predictions = np.array(predictions[q_id_index])

        # 10 sample docs for a round
        one_sample = []  # list of 10 dicts
        for i in range(10):  # different random restarts of the trigger
            orig_rank = int(doc_ranks[i])
            doc_index = doc_indexes[i]
            doc = sample_docs[i]

            trigger_token_ids = np.random.randint(low=999,
                                                  high=total_vocab_size, size=trigger_token_length)
            trigger_words = tokenizer.decode(trigger_token_ids)
            print("initial trigger words: " + trigger_words)
            # get initial loss for trigger
            rerank_model.model.zero_grad()
            loss = get_new_score(rerank_model, doc, query,
                                 trigger_token_ids, tokenizer)

            print('initial trigger loss:', loss)
            best_loss = loss  # get the loss of doc_index
            new_predictions = get_ranking_scores(
                orig_predictions, doc_index, loss)
            relative_changes = get_accuracy(
                doc_index, orig_predictions, new_predictions, direction)
            print("relative changes of doc: ", relative_changes)

            for _ in range(10):  # this many updates of the entire trigger sequence
                averaged_grad = get_average_grad_for_docstart(
                    loss, trigger_token_ids, trigger_start_position)  # shape of (5, 768)

                if direction == 'demotion':
                    cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                        averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=False)

                    trigger_token_ids = get_best_candidates_topdoc_docstart(
                        rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1)
                
                if direction == 'promotion':
                    cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                        averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=True)

                    trigger_token_ids = get_best_candidates_lastdoc_docstart(
                        rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1)

                trigger_words = tokenizer.decode(trigger_token_ids)
                loss = get_new_score(rerank_model, doc, query,
                                     trigger_token_ids, tokenizer)
                print("current trigger: ", trigger_words)
                print("current loss:", loss)
                
                new_predictions = get_ranking_scores(
                    orig_predictions, doc_index, loss)
                relative_changes = get_accuracy(
                    doc_index, orig_predictions, new_predictions, direction)
                
                if direction == 'demotion':
                    if loss < best_loss:
                        best_loss = loss
                    else:
                        break
                if direction == 'promotion':
                    if loss > best_loss:
                        best_loss = loss
                    else:
                        break
            print("final trigger words:" + trigger_words)
            trigger_info = {"original_rank": orig_rank,
                            "trigger_words": trigger_words,
                            "trigger_token_ids": trigger_token_ids.tolist(), "relative_changes": relative_changes}
            sample_dict = {doc_index: trigger_info}
            print(f'sample dict for {i}:\n', sample_dict)
            one_sample.append(sample_dict)
        samples.append(one_sample)
    query_record['samples'] = samples

    return query_record


def sample_docs(dataset='msmarco_p'):
    import csv
    project_dir = Path(__file__).parent.absolute()
    record_dir = project_dir / \
        f'Results/{dataset}/fold_1/rank_bert/multiple_docs'
    # get q_id list
    if dataset == 'msmarco_p':
        q_id_path = project_dir / 'Datasets/src/msmarco_p/queries_dev_978.txt'
        with open(q_id_path, 'r')as f:
            reader = csv.reader(f, delimiter='\t')
            q_ids = [row[0] for row in reader]

    if dataset == 'clueweb09':
        q_ids = range(0, 200)
    print('q id length:', len(q_ids))

    rerank_model = init_rerank(dataset, 'bert', 'fold_1')
    # for each query
    A = []
    B = []
    for ind, q_id in enumerate(q_ids[800:]):
        offset = 800
        q_id_index = ind + offset
        q_id = str(q_id)

        # 1.sample 10 docs in range [1,49] for demotion, 5 random seeds
        demotion_record = run_model(q_id_index,
                                    q_id=q_id, rerank_model=rerank_model, direction='demotion', dataset=dataset)
        A.append(demotion_record)
        if (ind+1) % 10 == 0:
            with open(record_dir/'demotion'/f'{(q_id_index+1)/10}.json', 'w')as f:
                json.dump(A, f)
            A = []
        # 2.sample 10 docs in range [50,len(top)] for promotion, 5 random seeds
        promotion_record = run_model(
            q_id_index, q_id=q_id, rerank_model=rerank_model, direction='promotion')
        B.append(promotion_record)
        if (ind+1) % 10 == 0:
            with open(record_dir/'promotion'/f'{(q_id_index+1)/10}.json', 'w')as f:
                json.dump(B, f)
            B = []


def global_instance_clueweb09():
    project_dir = Path(__file__).parent.absolute()
    # 1.demotion
    demotion_path1 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/demotion/*.json'
    demotion_files = [f for f in glob.glob(demotion_path1)]

    demotion_nrc = []
    for file in demotion_files:
        with open(file, 'r')as f:
            record = json.load(f)
        for sample in record['samples']:  # 5 random seed
            for d in sample:  # 10 records for seed
                for v in d.values():
                    demotion_nrc.append(v['relative_changes'])
    print('demotion length:', len(demotion_nrc))
    de_dic = {'direction': ['demotion'] *
              len(demotion_nrc), 'normalized rank shift': demotion_nrc, 'class': ['local attack']*len(demotion_nrc)}
    demotion_df = pd.DataFrame(data=de_dic)

    # 2.promotion
    promotion_path1 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/promotion/*.json'
    promotion_files = [f for f in glob.glob(promotion_path1)]

    promotion_nrc = []
    for file in promotion_files:
        with open(file, 'r')as f:
            record = json.load(f)
        for sample in record['samples']:  # 5 random seed
            for d in sample:  # 10 records for seed
                for v in d.values():
                    promotion_nrc.append(v['relative_changes'])
    print('promotion length:', len(promotion_nrc))
    pro_dic = {'direction': [
        'promotion']*len(promotion_nrc), 'normalized rank shift': promotion_nrc, 'class': ['local attack']*len(promotion_nrc)}
    promotion_df = pd.DataFrame(data=pro_dic)

    # 3. demotion_global
    demotino_path2 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/demotion_global/*.json'
    demotion_global_files = [f for f in glob.glob(demotino_path2)]

    demotion_global_nrc = []
    for file in demotion_global_files:
        with open(file, 'r')as f:
            record = json.load(f)
        demotion_global_nrc.extend(record[0][-1]['relative_changes'])
        # demotion_global_nrc.append(np.mean(record[0][-1]['relative_changes']))
    print('demotion length:', len(demotion_global_nrc))
    de_global_dic = {'direction': ['demotion'] *
                     len(demotion_global_nrc), 'normalized rank shift': demotion_global_nrc, 'class': ['global attack']*len(demotion_global_nrc)}
    demotion_global_df = pd.DataFrame(data=de_global_dic)

    # 4. promotion_global
    promotion_path2 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/promotion_global/*.json'
    promotion_global_files = [f for f in glob.glob(promotion_path2)]

    promotion_global_nrc = []
    for file in promotion_global_files:
        with open(file, 'r')as f:
            record = json.load(f)
        promotion_global_nrc.extend(record[0][-1]['relative_changes'])
        # promotion_global_nrc.append(np.mean(record[0][-1]['relative_changes']))
    print('promotion length:', len(promotion_global_nrc))
    pro_global_dic = {'direction': [
        'promotion']*len(promotion_global_nrc), 'normalized rank shift': promotion_global_nrc, 'class': ['global attack']*len(promotion_global_nrc)}
    promotion_global_df = pd.DataFrame(data=pro_global_dic)

    # 5. demotion_random
    demotion_path3 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/demotion_random/*.json'
    demotion_random_files = [f for f in glob.glob(demotion_path3)]

    demotion_random_nrc = []
    for file in demotion_random_files:
        with open(file, 'r')as f:
            record = json.load(f)
            for r in record:
                for sample in r['samples']:  # 5 random seed
                    for d in sample:  # 10 records for seed
                        for v in d.values():
                            demotion_random_nrc.append(v['relative_changes'])
    print('demotion length:', len(demotion_random_nrc))
    de_random_dic = {'direction': ['demotion'] *
                     len(demotion_random_nrc), 'normalized rank shift': demotion_random_nrc, 'class': ['random']*len(demotion_random_nrc)}
    demotion_random_df = pd.DataFrame(data=de_random_dic)

    # 6. promotion_random
    promotion_path3 = project_dir / 'Results/clueweb09/fold_1/rank_bert/multiple_docs/promotion_random/*.json'
    promotion_random_files = [f for f in glob.glob(promotion_path3)]

    promotion_random_nrc = []
    for file in promotion_random_files:
        with open(file, 'r')as f:
            record = json.load(f)
            for r in record:
                for sample in r['samples']:  # 5 random seed
                    for d in sample:  # 10 records for seed
                        for v in d.values():
                            promotion_random_nrc.append(v['relative_changes'])
    print('promotion length:', len(promotion_random_nrc))
    pro_random_dic = {'direction': [
        'promotion']*len(promotion_random_nrc), 'normalized rank shift': promotion_random_nrc, 'class': ['random']*len(promotion_random_nrc)}
    promotion_random_df = pd.DataFrame(data=pro_random_dic)

    # dataframe and plot
    df = pd.concat([demotion_random_df, promotion_random_df, demotion_global_df, promotion_global_df, demotion_df, promotion_df
                    ], ignore_index=True)

    return df

def global_instance_msmarco():
    project_dir = Path(__file__).parent.absolute()
    # 1.demotion
    demotion_path1 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/demotion/*.json'
    demotion_files = [f for f in glob.glob(demotion_path1)]

    demotion_nrc = []
    for file in demotion_files:
        with open(file, 'r')as f:
            records = json.load(f)
        for record in records:
            for sample in record['samples']:  # 5 random seed
                for d in sample:  # 10 records for seed
                    for v in d.values():
                        demotion_nrc.append(v['relative_changes'])
    print('demotion length:', len(demotion_nrc))
    de_dic = {'direction': ['demotion'] *
              len(demotion_nrc), 'normalized rank shift': demotion_nrc, 'class': ['local attack']*len(demotion_nrc)}
    demotion_df = pd.DataFrame(data=de_dic)

    # 2.promotion
    promotion_path1 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/promotion/*.json'
    promotion_files = [f for f in glob.glob(promotion_path1)]

    promotion_nrc = []
    for file in promotion_files:
        with open(file, 'r')as f:
            records = json.load(f)
        for record in records:
            for sample in record['samples']:  # 5 random seed
                for d in sample:  # 10 records for seed
                    for v in d.values():
                        promotion_nrc.append(v['relative_changes'])
    print('promotion length:', len(promotion_nrc))
    pro_dic = {'direction': [
        'promotion']*len(promotion_nrc), 'normalized rank shift': promotion_nrc, 'class': ['local attack']*len(promotion_nrc)}
    promotion_df = pd.DataFrame(data=pro_dic)

    # 3. demotion_global
    demotion_path2 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/demotion_global/*.json'
    demotion_global_files = [f for f in glob.glob(demotion_path2)]

    demotion_global_nrc = []
    for file in demotion_global_files:
        with open(file, 'r')as f:
            record = json.load(f)
        demotion_global_nrc.extend(record[0][-1]['relative_changes'])
        # demotion_global_nrc.append(np.mean(record[0][-1]['relative_changes']))
    print('demotion length:', len(demotion_global_nrc))
    de_global_dic = {'direction': ['demotion'] *
                     len(demotion_global_nrc), 'normalized rank shift': demotion_global_nrc, 'class': ['global attack']*len(demotion_global_nrc)}
    demotion_global_df = pd.DataFrame(data=de_global_dic)

    # 4. promotion_global
    promotion_path2 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/promotion_global/*.json'
    promotion_global_files = [f for f in glob.glob(promotion_path2)]

    promotion_global_nrc = []
    for file in promotion_global_files:
        with open(file, 'r')as f:
            record = json.load(f)
        promotion_global_nrc.extend(record[0][-1]['relative_changes'])
        # promotion_global_nrc.append(np.mean(record[0][-1]['relative_changes']))
    print('promotion length:', len(promotion_global_nrc))
    pro_global_dic = {'direction': [
        'promotion']*len(promotion_global_nrc), 'normalized rank shift': promotion_global_nrc, 'class': ['global attack']*len(promotion_global_nrc)}
    promotion_global_df = pd.DataFrame(data=pro_global_dic)

    # 5. demotion_random
    demotion_path3 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/demotion_random/*.json'
    demotion_random_files = [f for f in glob.glob(demotion_path3)]

    demotion_random_nrc = []
    for file in demotion_random_files:
        with open(file, 'r')as f:
            record = json.load(f)
            for r in record:
                for sample in r['samples']:  # 5 random seed
                    for d in sample:  # 10 records for seed
                        for v in d.values():
                            demotion_random_nrc.append(v['relative_changes'])
    print('demotion length:', len(demotion_random_nrc))
    de_random_dic = {'direction': ['demotion'] *
                     len(demotion_random_nrc), 'normalized rank shift': demotion_random_nrc, 'class': ['random']*len(demotion_random_nrc)}
    demotion_random_df = pd.DataFrame(data=de_random_dic)

    # 6. promotion_random
    promotion_path3 = project_dir / 'Results/msmarco_p/fold_1/rank_bert/multiple_docs/promotion_random/*.json'
    promotion_random_files = [f for f in glob.glob(promotion_path3)]

    promotion_random_nrc = []
    for file in promotion_random_files:
        with open(file, 'r')as f:
            record = json.load(f)
            for r in record:
                for sample in r['samples']:  # 5 random seed
                    for d in sample:  # 10 records for seed
                        for v in d.values():
                            promotion_random_nrc.append(v['relative_changes'])
    print('promotion length:', len(promotion_random_nrc))
    pro_random_dic = {'direction': [
        'promotion']*len(promotion_random_nrc), 'normalized rank shift': promotion_random_nrc, 'class': ['random']*len(promotion_random_nrc)}
    promotion_random_df = pd.DataFrame(data=pro_random_dic)

    # dataframe and plot
    df = pd.concat([demotion_random_df, promotion_random_df, demotion_global_df, promotion_global_df, demotion_df, promotion_df
                    ], ignore_index=True)

    return df

def plot_global_instance_box():
    df_clueweb09 = global_instance_clueweb09()
    df_msmarco = global_instance_msmarco()

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    palette = sns.cubehelix_palette(8, start=0.5, rot=-0.85, gamma=0.8)

    # sns.boxplot(x="direction", y="normalized rank shift", hue='class', ax=ax1,
    #            data=df_clueweb09, palette="Set3", showfliers=False)
    sns.barplot(x="direction", y="normalized rank shift", hue='class',
                ax=ax1, ci='sd',  # capsize=.2,
                data=df_clueweb09, palette=palette[0:5:2])
    for ind, container in enumerate(ax1.containers):
        if ind == 0 or ind == 3:
            continue
        ax1.bar_label(container, fmt='%.2f', label_type='center', fontsize=15)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.set_ylim([0, 1])
    ax1.get_yaxis().set_visible(True)

    ax1.set_xticklabels(['rank demotion', 'rank promotion'], fontsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.set_xlabel('ClueWeb09', fontsize=17)
    ax1.set_ylabel('normalized rank shift', fontsize=19)
    ax1.legend(loc='upper left', bbox_to_anchor=(8.8/10, 1.05), fontsize=13)

    ax2.set_zorder(-1)
    sns.barplot(x="direction", y="normalized rank shift", hue='class',
                ax=ax2, ci='sd',  # capsize=.2,
                data=df_msmarco, palette=palette[0:5:2])
    for ind, container in enumerate(ax2.containers):
        if ind == 0 or ind == 3:
            continue
        ax2.bar_label(container, fmt='%.2f', label_type='center', fontsize=15)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(True)
    ax2.set_ylim([0, 1])
    ax2.get_yaxis().set_visible(False)
    # ax2.set_xticks(x)
    ax2.set_xticklabels(['rank demotion', 'rank promotion'], fontsize=15)
    ax2.set_xlabel('TREC-DL', fontsize=17)
    ax2.legend_.remove()

    project_dir = Path(__file__).parent.absolute()
    plot_path = project_dir / 'Results/paper_plot'
    plt.savefig(plot_path/'multiple_docs_barplot.eps',
                format='eps', bbox_inches='tight')
    plt.savefig(plot_path/'multiple_docs_barplot.png',
                bbox_inches='tight')  # , bbox_inches='tight'


def sample_docs_batch_demotion(dataset='clueweb09'):
    random_seed = 3
    seed_everything(random_seed, workers=True)

    import attack_global_merge
    project_dir = Path(__file__).parent.absolute()
    rerank_model = init_rerank(dataset, 'bert', 'fold_1')
    if dataset == 'clueweb09':
        trigger_token_length = 5
    if dataset == 'msmarco_p':
        trigger_token_length = 3

    if dataset == 'clueweb09':
        rank_file = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query/200_rank.json'
        id_pair_file = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query/id_pair.json'
        with open(id_pair_file, 'r')as f:
            id_pair = json.load(f)
        query_ids = [pair[0] for pair in id_pair]
        range_num = 50  # 200//4
    if dataset == 'msmarco_p':
        rank_file = project_dir / 'Results/msmarco_p/fold_1/rank_bert/query_rank/ranks.json'
        id_pair_file = project_dir / 'Results/msmarco_p/fold_1/rank_bert/id_pair.json'
        with open(id_pair_file, 'r')as f:
            id_pair = json.load(f)
        query_ids = [pair[0] for pair in id_pair]
        range_num = 100  # 978//4

    with open(rank_file, 'r')as f:
        ranks = json.load(f)

    # 10 random seed
    for random_number in range(8, 30):

        """ demotion"""
        print('start demotion')
        batch_ids = []
        queries, batch_docs = [], []

        top_file_number = [len(row) for row in ranks]

        # get file batch_ids: q_id, doc_index
        for ind in range(range_num):
            length = min(top_file_number[ind*4: ind*4+4])
            doc_ranks = np.random.randint(
                low=0, high=length//2, size=4)  # size = batch_size

            q_ids = [query_ids[ind*4+0], query_ids[ind*4+1],
                     query_ids[ind*4+2], query_ids[ind*4+3]]
            for i, q_id in enumerate(q_ids):
                doc_rank = doc_ranks[i]
                doc_index = ranks[ind*4+i][doc_rank]
                batch_ids.append([q_id, doc_index])

                queries.append(rerank_model.queries[q_id])

                rerank_model.InferenceDataset.__init_q_docs__(
                    q_id, rerank_model.queries[q_id])
                top_docs = rerank_model.InferenceDataset.top_docs
                batch_docs.append(top_docs[doc_index])
        print('len of:', len(batch_ids))

        # define parameters and call function
        save_path = project_dir / f'Results/{dataset}/fold_1/rank_bert/multiple_docs/demotion_global/random_number{random_number}.json'
        attack_global_merge.global_merge(dataset=dataset, doc_position='topdoc', trigger_token_length=trigger_token_length,  # topdoc means demotion
                                         load=False, batch_ids=batch_ids, queries=queries, batch_docs=batch_docs, save_path=save_path)


def sample_docs_batch_promotion(dataset='clueweb09'):
    random_seed = 2
    seed_everything(random_seed, workers=True)

    import attack_global_merge
    rerank_model = init_rerank(dataset, 'bert', 'fold_1')
    project_dir = Path(__file__).parent.absolute()
    if dataset == 'clueweb09':
        trigger_token_length = 5
    if dataset == 'msmarco_p':
        trigger_token_length = 3

    if dataset == 'clueweb09':
        rank_file = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query/200_rank.json'
        id_pair_file = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query/id_pair.json'
        with open(id_pair_file, 'r')as f:
            id_pair = json.load(f)
        query_ids = [pair[0] for pair in id_pair]
        range_num = 50  # 200//4
    if dataset == 'msmarco_p':
        rank_file = project_dir / 'Results/msmarco_p/fold_1/rank_bert/query_rank/ranks.json'
        id_pair_file = project_dir / 'Results/msmarco_p/fold_1/rank_bert/id_pair.json'
        with open(id_pair_file, 'r')as f:
            id_pair = json.load(f)
        query_ids = [pair[0] for pair in id_pair]
        range_num = 100  # 978//4

    with open(rank_file, 'r')as f:
        ranks = json.load(f)
# 10 random seed
    for random_number in range(1, 20):
        """ promotion"""
        print('start promotion')
        batch_ids = []
        queries, batch_docs = [], []

        top_file_number = [len(row) for row in ranks]

        # get file batch_ids: q_id, doc_index
        for ind in range(range_num):
            # define min length
            length = min(top_file_number[ind*4: ind*4+4])
            doc_ranks = np.random.randint(
                low=length//2, high=length, size=4)  # size = batch_size

            q_ids = [query_ids[ind*4+0], query_ids[ind*4+1],
                     query_ids[ind*4+2], query_ids[ind*4+3]]
            for i, q_id in enumerate(q_ids):
                doc_rank = doc_ranks[i]
                doc_index = ranks[ind*4+i][doc_rank]
                batch_ids.append([q_id, doc_index])
                queries.append(rerank_model.queries[q_id])
                rerank_model.InferenceDataset.__init_q_docs__(
                    q_id, rerank_model.queries[q_id])
                top_docs = rerank_model.InferenceDataset.top_docs
                batch_docs.append(top_docs[doc_index])
        print('len of:', len(batch_ids))

        # define parameters and call function
        save_path = project_dir / f'Results/{dataset}/fold_1/rank_bert/multiple_docs/promotion_global/random_number{random_number}.json'
        attack_global_merge.global_merge(dataset=dataset, doc_position='lastdoc', trigger_token_length=trigger_token_length,
                                         load=False, batch_ids=batch_ids, queries=queries, batch_docs=batch_docs, save_path=save_path)


def random_attack(dataset='clueweb09'):
    import csv
    project_dir = Path(__file__).parent.absolute()
    record_dir = project_dir / \
        f'Results/{dataset}/fold_1/rank_bert/multiple_docs'
    # get q_id list
    if dataset == 'msmarco_p':
        q_id_path = project_dir / 'Datasets/src/msmarco_p/queries_dev_978.txt'
        with open(q_id_path, 'r')as f:
            reader = csv.reader(f, delimiter='\t')
            q_ids = [row[0] for row in reader]

    if dataset == 'clueweb09':
        q_ids = range(1, 201)
    print('q id length:', len(q_ids))

    rerank_model = init_rerank(dataset, 'bert', 'fold_1')
    # for each query
    A = []
    B = []
    for ind, q_id in enumerate(q_ids):
        offset = 0
        q_id_index = ind + offset
        q_id = str(q_id)

        # 1.sample 10 docs in range e.g [1,49] for demotion, 5 random seeds
        demotion_record = _random_attack(q_id_index,
                                         q_id=q_id, rerank_model=rerank_model, dataset=dataset, direction='demotion')
        A.append(demotion_record)
        if (ind+1) % 20 == 0:
            with open(record_dir/'demotion_random'/f'{(q_id_index+1)/20}.json', 'w')as f:
                json.dump(A, f)
            A = []
        # 2.sample 10 docs in range e.g [50,len(top)] for promotion, 5 random seeds
        promotion_record = _random_attack(
            q_id_index, q_id=q_id, rerank_model=rerank_model, dataset=dataset, direction='promotion')
        B.append(promotion_record)
        if (ind+1) % 20 == 0:
            with open(record_dir/'promotion_random'/f'{(q_id_index+1)/20}.json', 'w')as f:
                json.dump(B, f)
            B = []


def _random_attack(q_id_index, q_id, rerank_model, dataset='clueweb09', direction='demotion'):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    if dataset == 'clueweb09':
        trigger_token_length = 5
    if dataset == 'msmarco_p':
        trigger_token_length = 3
    seed_everything(random_seed, workers=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    project_dir = Path(__file__).parent.absolute()
    if dataset == 'clueweb09':
        rank_path = project_dir / 'Results/clueweb09/fold_1/rank_bert/200_query'
        with open(rank_path/'200_rank.json', 'r')as f:
            ranks = json.load(f)
        with open(rank_path/'200_prediction.json', 'r')as f:
            predictions = json.load(f)
    if dataset == 'msmarco_p':
        rank_path = project_dir / 'Results/msmarco_p/fold_1/rank_bert/query_rank'
        with open(rank_path/'ranks.json', 'r')as f:
            ranks = json.load(f)
        with open(rank_path/'predictions.json', 'r')as f:
            predictions = json.load(f)

    query_record = {'q_id': q_id, 'query': rerank_model.queries[q_id]}
    # 5 random seeds
    samples = []
    for x in range(5):
        query = rerank_model.queries[q_id]
        rerank_model.InferenceDataset.__init_q_docs__(
            q_id, rerank_model.queries[q_id])
        top_docs = rerank_model.InferenceDataset.top_docs
        length = min(100, len(top_docs))

        if direction == 'demotion':
            doc_ranks = np.random.randint(low=0, high=length//2, size=10)
        if direction == 'promotion':
            doc_ranks = np.random.randint(
                low=length//2, high=length, size=10)

        doc_indexes = [ranks[q_id_index][r]
                       for r in doc_ranks]  # 10 random docs
        sample_docs = [top_docs[index] for index in doc_indexes]
        # get trigger start position --> for gradient collecting
        # e.g include[CLS] and [SEP] [101, 9019, 4391, 102]
        query_token_id = tokenizer.encode(query, add_special_tokens=False)
        trigger_start_position = len(query_token_id)+2
        # rerank_model._init_query(q_id, rank_scores=True)
        orig_predictions = np.array(predictions[q_id_index])

        # 10 sample docs for a round
        one_sample = []  # list of 10 dicts
        for i in range(10):  # different random restarts of the trigger
            orig_rank = int(doc_ranks[i])
            doc_index = doc_indexes[i]
            doc = sample_docs[i]

            trigger_token_ids = np.random.randint(low=999,
                                                  high=total_vocab_size, size=trigger_token_length)
            trigger_words = tokenizer.decode(trigger_token_ids)
            print("initial trigger words: " + trigger_words)
            # get initial loss for trigger
            rerank_model.model.zero_grad()
            loss = get_new_score(rerank_model, doc, query,
                                 trigger_token_ids, tokenizer)

            print('initial trigger loss:', loss)
            best_loss = loss  # get the loss of doc_index
            new_predictions = get_ranking_scores(
                orig_predictions, doc_index, loss)
            relative_changes = get_accuracy(
                doc_index, orig_predictions, new_predictions, direction)
            print("relative changes of doc: ", relative_changes)
            trigger_info = {"original_rank": orig_rank,
                            "trigger_words": trigger_words,
                            "trigger_token_ids": trigger_token_ids.tolist(),
                            "relative_changes": relative_changes}
            sample_dict = {doc_index: trigger_info}
            one_sample.append(sample_dict)
        samples.append(one_sample)
    query_record['samples'] = samples

    return query_record


if __name__ == '__main__':
    sample_docs(dataset='msmarco_p')  
    sample_docs(dataset='clueweb09')

    sample_docs_batch_demotion(dataset='clueweb09')
    sample_docs_batch_promotion(dataset='clueweb09')

    random_attack(dataset='clueweb09')
    random_attack(dataset='msmarco_p')

    global_instance_clueweb09()
    global_instance_msmarco()

    plot_global_instance_box()
