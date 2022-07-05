"""
This file can reproduct Figure 1, 3, 4, 5 based on 
experiment results that is obtained from attack_trigger.length.py and 
attack_trigger_position.py.
"""
from transformers import BertTokenizer  # get related tokenizer
from ranklist import device
import json
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from collections import Counter
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

project_dir = Path(__file__).parent.absolute()

def plot_length_box():
    
    record_path = project_dir / 'Results/clueweb09/fold_1/rank_bert/length_record'
    plot_path = project_dir / 'Results/paper_plot'
    # get average performance for topdoc
    topdoc_files = [f'topdoc_trigger{length}_record.json' for length in [
        1, 3, 5, 7, 9, 13, 20]]
    topdoc_path = [record_path/file for file in topdoc_files]
    print('topdoc length', len(topdoc_path))

    # create dataframe for sns boxplot
    demotion_dict = {}
    for i, path in enumerate(topdoc_path):
        lengths = [1, 3, 5, 7, 9, 13, 20]
        nrs = []  # normalized relative changes

        with open(path, 'r')as f:
            trigger_record = json.load(f)

        for record in trigger_record:
            nrs.extend(record['trigger_loss_mean'])

        if i == 0:
            demotion_dict['length'] = [lengths[i]] * len(nrs)
            demotion_dict['normalized rank shift'] = nrs
            demotion_dict['direction'] = ['demotion'] * len(nrs)
        else:
            demotion_dict['length'].extend([lengths[i]] * len(nrs))
            demotion_dict['normalized rank shift'].extend(nrs)
            demotion_dict['direction'].extend(['demotion'] * len(nrs))

    demotion_df = pd.DataFrame(data=demotion_dict)

    # get performance for lastdoc
    lastdoc_files = [f'lastdoc_trigger{length}_record.json' for length in [
        1, 3, 5, 7, 9, 13, 20]]
    lastdoc_path = [record_path/file for file in lastdoc_files]
    print('lastdoc length', len(lastdoc_path))

    promotion_dict = {}
    for i, path in enumerate(lastdoc_path):
        lengths = [1, 3, 5, 7, 9, 13, 20]
        nrs = []  # normalized relative changes

        with open(path, 'r')as f:
            trigger_record = json.load(f)

        for record in trigger_record:
            nrs.extend(record['trigger_loss_mean'])

        if i == 0:
            promotion_dict['length'] = [lengths[i]] * len(nrs)
            promotion_dict['normalized rank shift'] = nrs
            promotion_dict['direction'] = ['promotion'] * len(nrs)
        else:
            promotion_dict['length'].extend([lengths[i]] * len(nrs))
            promotion_dict['normalized rank shift'].extend(nrs)
            promotion_dict['direction'].extend(['promotion'] * len(nrs))
    promotion_df = pd.DataFrame(data=promotion_dict)

    dic = pd.concat([demotion_df, promotion_df], ignore_index=True)
    df = pd.DataFrame(data=dic)

    plt.figure()

    palette = sns.color_palette(['#F1D77E', '#B1CE46'])
    ax = sns.boxplot(x="length", y="normalized rank shift", hue="direction",
                     data=df, palette=palette, showfliers=False)
    plt.yticks(fontsize=12)
    plt.xlabel('length', fontsize=18)
    plt.ylabel('normalized rank shift', fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig(plot_path/'length_boxplot_color1.eps',
                format='eps', bbox_inches='tight')
    plt.savefig(plot_path/'length_boxplot_color1.png', bbox_inches='tight')

def plot_position_box():
    record_path = project_dir / 'Results/clueweb09/fold_1/rank_bert/position_record'
    plot_path = project_dir / 'Results/paper_plot'
    # get performance for topdoc
    positions = ['random',  'middleplace',
                 'inplace_mingradient', 'inplace', 'topplace', 'lastplace']
    topdoc_files = [
        f'{position}_topdoc_trigger5_record.json' for position in positions]
    topdoc_path = [record_path/file for file in topdoc_files]
    print('topdoc length', len(topdoc_path))

    # create dataframe for sns boxplot
    demotion_dict = {}
    for i, path in enumerate(topdoc_path):
        x = ['random', 'middle',
             'min-grad', 'max-grad', 'start', 'end']
        nrs = []

        with open(path, 'r')as f:
            trigger_record = json.load(f)

        for record in trigger_record:
            nrs.extend(record['trigger_loss_mean'])

        if i == 0:
            demotion_dict['position'] = [x[i]] * len(nrs)
            demotion_dict['normalized rank shift'] = nrs
            demotion_dict['direction'] = ['demotion'] * len(nrs)
        else:
            demotion_dict['position'].extend([x[i]] * len(nrs))
            demotion_dict['normalized rank shift'].extend(nrs)
            demotion_dict['direction'].extend(['demotion'] * len(nrs))

    demotion_df = pd.DataFrame(data=demotion_dict)
    print('nrc length', len(nrs))
    print(demotion_df)

    # get performance for lastdoc
    positions = ['random',  'middleplace',
                 'inplace_mingradient', 'inplace', 'topplace', 'lastplace']
    lastdoc_files = [
        f'{position}_lastdoc_trigger5_record.json' for position in positions]
    lastdoc_path = [record_path/file for file in lastdoc_files]
    print('lastdoc length', len(lastdoc_path))

    promotion_dict = {}
    for i, path in enumerate(lastdoc_path):
        x = ['random', 'middle',
             'min-grad', 'max-grad', 'start', 'end']
        nrs = []

        with open(path, 'r')as f:
            trigger_record = json.load(f)

        for record in trigger_record:
            nrs.extend(record['trigger_loss_mean'])

        if i == 0:
            promotion_dict['position'] = [x[i]] * len(nrs)
            promotion_dict['normalized rank shift'] = nrs
            promotion_dict['direction'] = ['promotion'] * len(nrs)
        else:
            promotion_dict['position'].extend([x[i]] * len(nrs))
            promotion_dict['normalized rank shift'].extend(nrs)
            promotion_dict['direction'].extend(['promotion'] * len(nrs))
    promotion_df = pd.DataFrame(data=promotion_dict)

    dic = pd.concat([demotion_df, promotion_df], ignore_index=True)
    df = pd.DataFrame(data=dic)

    palette = sns.color_palette(['#F1D77E', '#B1CE46'])
    plt.figure(figsize=(8, 5))

    ax = sns.boxplot(x="position", y="normalized rank shift", hue="direction",
                     data=df, palette=palette, showfliers=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('position', fontsize=18)
    plt.ylabel('normalized rank shift', fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig(plot_path/'position_boxplot_color1.eps',
                format='eps', bbox_inches='tight')
    plt.savefig(plot_path/'position_boxplot_color1.png', bbox_inches='tight')

def heatmap_clueweb09(dataset=40, doc_position='topdoc'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if dataset == 40:
        record_path = project_dir / 'Results' / 'clueweb09' / \
            'fold_1' / 'rank_bert'
        if doc_position == 'topdoc':
            plot_path = record_path / 'plots2' / 'topdoc'
            with open(record_path/'topdoc_trigger5_record.json', 'r')as f:
                topdoc_trigger = json.load(f)
            with open(record_path/'batch_topdoc.json', 'r')as f:
                query_doc = json.load(f)
        if doc_position == 'lastdoc':
            plot_path = record_path / 'plots2' / 'lastdoc'
            with open(record_path/'lastdoc_trigger5_record.json', 'r')as f:
                topdoc_trigger = json.load(f)
            with open(record_path/'batch_lastdoc.json', 'r')as f:
                query_doc = json.load(f)

    elif dataset == 200:
        dir_path = project_dir / 'Results' / 'clueweb09' / \
            'fold_1' / 'rank_bert'
        record_path = dir_path / '200_query'   
        if doc_position == 'topdoc':
            plot_path = dir_path / 'plots2' / 'topdoc_200'
            with open(record_path/'topdoc_trigger5_record_200.json', 'r')as f:
                topdoc_trigger = json.load(f)
            with open(record_path/'batch_topdoc.json', 'r')as f:
                query_doc = json.load(f)
        if doc_position == 'lastdoc':
            plot_path = dir_path / 'plots2' / 'lastdoc_200'
            with open(record_path/'lastdoc_trigger5_record_200.json', 'r')as f:
                topdoc_trigger = json.load(f)
            with open(record_path/'batch_lastdoc.json', 'r')as f:
                query_doc = json.load(f)

    all_best_trigger_dict = []  # list of all best trigger info 
    all_best_unique_id = []  # list[int]
    for i in range(len(topdoc_trigger)): 
        dic = {}
        query_id = topdoc_trigger[i]['q_id']
        trigger_6_times = topdoc_trigger[i]['trigger_6_times']
        best_trigger = sorted(
            trigger_6_times, key=lambda d: d['relative_changes'])[-1]
        best_trigger_id = best_trigger['trigger_token_ids']  # list[int]
        all_best_unique_id += best_trigger_id
        # get best trigger info for every query
        dic['q_id'] = query_id
        dic['doc_index'] = topdoc_trigger[i]['doc_index']
        dic['trigger_words'] = best_trigger['trigger_words']
        dic['trigger_token_ids'] = best_trigger['trigger_token_ids']
        dic['relative_changes'] = best_trigger['relative_changes']
        all_best_trigger_dict.append(dic)

    # save best trigger 5 set in a json file
    if dataset == 40:
        save_dir = record_path/f'{doc_position}_trigger5_bestset.json'
    elif dataset == 200:
        save_dir = record_path/f'{doc_position}_trigger5_bestset_200.json'
    with open(save_dir, 'w')as f:
        json.dump(all_best_trigger_dict, f)

    # step 1. plot counts of most frequent triggers words
    all_best_id_count = Counter(
        all_best_unique_id).most_common()  # decending order
    if dataset == 40:
        most_frequent_count = [
            pair for pair in all_best_id_count if pair[1] > 1]  # list[(id,count)]
    elif dataset == 200:
        most_frequent_count = [
            pair for pair in all_best_id_count if pair[1] > 5]

    X_ticks = [tokenizer.convert_ids_to_tokens(
        pair[0]) for pair in most_frequent_count]
    x = np.arange(len(most_frequent_count))+1
    y = [pair[1] for pair in most_frequent_count]
    plt.figure(figsize=(14, 8))
    plt.bar(x, y, color='#82B0D2')
    plt.xticks(x, X_ticks, fontsize=12, rotation=30)
    plt.xlabel('the most frequently occuring adversarial tokens', fontsize=16)
    plt.ylabel('counts', fontsize=18)

    for a, b in zip(x, y):
        plt.text(a, b+0.1, b, ha='center', va='bottom')
    plt.savefig(plot_path/'Count_frequent_trigger.png',
                bbox_inches='tight')
    plt.savefig(plot_path/'Count_frequent_trigger.eps',
                bbox_inches='tight')
    plt.close()

    # step 2. plot all best triggers in a heatmap to observe token sequence
    matrix_token_id = np.array([t['trigger_token_ids']
                               for t in all_best_trigger_dict])  # shape(40,5)/(200,5)
    # as annotation shown on heatmap
    matrix_words = []
    for i in range(len(all_best_trigger_dict)):
        token_ids = all_best_trigger_dict[i]['trigger_token_ids']
        matrix_words.append(tokenizer.convert_ids_to_tokens(token_ids))
    matrix_words = np.array(matrix_words)
    matrix_color = np.zeros_like(matrix_token_id)
    
    for row in range(matrix_token_id.shape[0]):
        for col in range(matrix_token_id.shape[1]):
            # if matrix_token_id[row][col] not in frequent_trigger_color:
            if matrix_token_id[row][col] not in dict(most_frequent_count):
                matrix_color[row][col] = 0
            else:
                # matrix_color[row][col] = frequent_trigger_color[matrix_token_id[row][col]]
                d = dict(most_frequent_count)
                matrix_color[row][col] = d[matrix_token_id[row][col]]

    if dataset == 40:
        plt.figure(figsize=(5, 10))
    elif dataset == 200:
        plt.figure(figsize=(8, 30))
    sns.heatmap(matrix_color, cmap='YlGnBu', cbar_kws={"orientation": "horizontal"},
                annot=matrix_words, fmt='', annot_kws={'fontsize': 9})
    
    plt.xlabel('local adversarial tokens', fontsize=12)
    plt.ylabel('query items', fontsize=12)
    plt.yticks(np.arange(0.5, len(topdoc_trigger), 1),
               query_doc['queries'], fontsize=9, rotation=0)
    
    fig_path = plot_path/'Heatmap_frequent_count1.eps'
    fig_path2 = plot_path/'Heatmap_frequent_count1.png'
    plt.savefig(fig_path, bbox_inches='tight')  # format='eps'
    plt.savefig(fig_path2, bbox_inches='tight')  # format='eps'
    plt.close()

def count_msmarco(doc_position='lastdoc'):
    # count frequent triggers
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    record_path = project_dir / 'Results' / 'msmarco_p' / \
        'fold_1' / 'rank_bert'
    plot_path = record_path / 'plots2'

    if doc_position == 'topdoc':
        with open(record_path/'id_pair.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'topdoc_trigger3_record.json', 'r')as f:
            doc_trigger = json.load(f)
        with open(record_path/'batch_topdoc_trigger3_record.json', 'r')as f:
            batch_record = json.load(f)
    if doc_position == 'lastdoc':
        with open(record_path/'id_pair_lastdoc.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'lastdoc_trigger3_record.json', 'r')as f:
            doc_trigger = json.load(f)
        with open(record_path/'batch_lastdoc_trigger3_record.json', 'r')as f:
            batch_record = json.load(f)

    # step 1. get best trigger set
    all_best_trigger_dict = []  
    all_best_unique_id = []  # list[int]
    for i in range(len(doc_trigger)):  
        dic = {}
        query_id = doc_trigger[i]['q_id']
        trigger_6_times = doc_trigger[i]['trigger_6_times']
        best_trigger = sorted(
            trigger_6_times, key=lambda d: d['relative_changes'])[-1]
        best_trigger_id = best_trigger['trigger_token_ids'] 
        all_best_unique_id += best_trigger_id
        # get best trigger info for every query
        dic['q_id'] = query_id
        dic['doc_index'] = doc_trigger[i]['doc_index']
        dic['trigger_words'] = best_trigger['trigger_words']
        dic['trigger_token_ids'] = best_trigger['trigger_token_ids']
        dic['relative_changes'] = best_trigger['relative_changes']
        all_best_trigger_dict.append(dic)
    # save best trigger set in json file
    save_dir = record_path/f'{doc_position}_trigger3_bestset.json'
    with open(save_dir, 'w')as f:
        json.dump(all_best_trigger_dict, f)

    # print batch triggers
    batch_words = [batch_list[-1]['trigger_words']
                   for batch_list in batch_record]
    print('batch words:', batch_words)

    # step 2. plot counts of frequent triggers
    all_best_id_count = Counter(
        all_best_unique_id).most_common()
    if doc_position == 'topdoc':
        most_frequent_count = [
            pair for pair in all_best_id_count if pair[1] > 25]
    if doc_position == 'lastdoc':
        most_frequent_count = [
            pair for pair in all_best_id_count if pair[1] > 12]
    X_ticks = [tokenizer.convert_ids_to_tokens(
        pair[0]) for pair in most_frequent_count]
    with open(plot_path/'lasdoc_bias_tokens.json', 'w')as f:
        json.dump(X_ticks, f)
    x = np.arange(len(most_frequent_count))+1
    y = [pair[1] for pair in most_frequent_count]
    with open(plot_path/'lasdoc_bias_tokens_counts.json', 'w')as f:
        json.dump(y, f)
    plt.figure(figsize=(14, 8))
    plt.bar(x, y, color='#82B0D2')
    plt.xticks(x, X_ticks, fontsize=12, rotation=30)
    plt.xlabel('the most frequently occuring adversarial tokens', fontsize=16)
    plt.ylabel('counts', fontsize=18)
    for a, b in zip(x, y):
        plt.text(a, b+0.1, b, ha='center', va='bottom')
    plt.savefig(
        plot_path/f'Count_frequent_trigger_{doc_position}.png', bbox_inches='tight')
    plt.savefig(
        plot_path/f'Count_frequent_trigger_{doc_position}.eps', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_length_box()
    plot_position_box()
    heatmap_clueweb09(dataset=40,doc_position='topdoc')
    count_msmarco(doc_position='topdoc')