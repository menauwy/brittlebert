from pathlib import Path
import matplotlib.pylab as plt
import json
from collections import Counter
from transformers import BertTokenizer
import numpy as np
import seaborn as sns

project_dir = Path('/home/wang/attackrank')
record_path = project_dir / 'Results' / 'clueweb09' / \
    'fold_1' / 'rank_bert' / 'length_record'
with open(record_path/'topdoc_trigger1_record.json', 'r')as f:
    record_1 = json.load(f)
with open(record_path/'topdoc_trigger3_record.json', 'r')as f:
    record_3 = json.load(f)
with open(record_path/'topdoc_trigger5_record.json', 'r')as f:
    record_5 = json.load(f)
with open(record_path/'topdoc_trigger7_record.json', 'r')as f:
    record_7 = json.load(f)
with open(record_path/'topdoc_trigger9_record.json', 'r')as f:
    record_9 = json.load(f)
with open(record_path/'topdoc_trigger13_record.json', 'r')as f:
    record_13 = json.load(f)
with open(record_path/'topdoc_trigger20_record.json', 'r')as f:
    record_20 = json.load(f)


def plot():
    # get all best relative changes into list[list] shape(5*40)
    all_changes = []
    for record in (record_1, record_3, record_5, record_7, record_9, record_13, record_20):
        best_changes = []
        for i, res in enumerate(record):
            trigger_6_times = res['trigger_6_times']
            best_trigger = sorted(
                trigger_6_times, key=lambda d: d['relative_changes'])[-1]
            best_changes.append(best_trigger['relative_changes'])
        all_changes.append(best_changes)

    plt.figure(figsize=(10, 6))
    colors = ['grey', 'slateblue', 'seagreen',
              'darkorange', 'hotpink', 'indianred', 'darkviolet']
    labels = [1, 3, 5, 7, 9, 13, 20]
    for i in range(len(colors)):
        plt.plot(all_changes[i], color=colors[i],
                 lw=2, label=f'{labels[i]} triggers')
    plt.xlabel('query index', fontsize=12)
    plt.ylabel('relative changes', fontsize=12)
    plt.title('Comparison of different trigger length on topdoc', fontsize=14)
    plt.legend()
    plt.savefig(record_path/'comparison_trigger_length.png')


def draw_heatmap():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # get file list in the folder
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    plot_path = record_path / 'length_record'
    file_path = plot_path

    # find path for json file
    lengthes = [1, 3, 5, 7, 9, 13, 20]  # add random later
    for length in lengthes:
        file = file_path / f'topdoc_trigger{length}_record.json'
        with open(file, 'r')as f:
            topdoc_trigger = json.load(f)

        # 1. get unique id & best trigger info
        all_best_trigger_dict = []  # list of all best trigger info 40*5
        all_best_unique_id = []  # list[int]
        for i in range(len(topdoc_trigger)):  # 40 queries
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

        # 2. plot counts of most frequent triggers words
        all_best_id_count = Counter(
            all_best_unique_id).most_common()  # decending order
        if length > 13:
            most_frequent_count = [
                pair for pair in all_best_id_count if pair[1] > 3]
        elif length > 5:
            most_frequent_count = [
                pair for pair in all_best_id_count if pair[1] > 2]
        else:
            most_frequent_count = [
                pair for pair in all_best_id_count if pair[1] > 1]  # list[(id,count)]
        print('most_frequent_count:', most_frequent_count)
        X_ticks = [tokenizer.convert_ids_to_tokens(
            pair[0]) for pair in most_frequent_count]
        x = np.arange(len(most_frequent_count))+1
        y = [pair[1] for pair in most_frequent_count]

        plt.figure(figsize=(10, 8))
        plt.bar(x, y, alpha=0.8)
        plt.xticks(x, X_ticks, size='small', rotation=30)
        plt.xlabel('most frequent trigger', fontsize=12)
        plt.ylabel('counts')
        plt.title(
            f'Counts of most frequent trigger words with {length}_trigger_length', fontsize=14)
        for a, b in zip(x, y):
            plt.text(a, b+0.1, b, ha='center', va='bottom')
        plt.savefig(plot_path/f'{length}_Count_frequent_trigger.png')
        plt.close()

        # 3. plot all best triggers in a heatmap, observe combinations
        matrix_token_id = np.array([t['trigger_token_ids']
                                   for t in all_best_trigger_dict])  # shape(40,5)/(200,5)
        # as annotation shown on heatmap
        matrix_words = []
        for i in range(len(all_best_trigger_dict)):
            token_ids = all_best_trigger_dict[i]['trigger_token_ids']
            matrix_words.append(tokenizer.convert_ids_to_tokens(token_ids))
        matrix_words = np.array(matrix_words)
        #print('matrix_words:', matrix_words)

        matrix_color = np.zeros_like(matrix_token_id)
        """
        frequent_trigger_color = {}  # dict{id:color_num} # most frequent has the largest num
        for i, tup in enumerate(most_frequent_count):
            frequent_trigger_color[tup[0]] = len(most_frequent_count) - i
        print('trigger color:', frequent_trigger_color)
        """
        for row in range(matrix_token_id.shape[0]):
            for col in range(matrix_token_id.shape[1]):
                # if matrix_token_id[row][col] not in frequent_trigger_color:
                if matrix_token_id[row][col] not in dict(most_frequent_count):
                    matrix_color[row][col] = 0
                else:
                    #matrix_color[row][col] = frequent_trigger_color[matrix_token_id[row][col]]
                    d = dict(most_frequent_count)
                    matrix_color[row][col] = d[matrix_token_id[row][col]]
        if length > 5:
            plt.figure(figsize=(6+length-5, 8))
        else:
            plt.figure(figsize=(5, 8))
        sns.heatmap(matrix_color, cmap='YlGnBu', cbar_kws={"orientation": "horizontal"},
                    annot=matrix_words, fmt='', annot_kws={'fontsize': 9})
        plt.xlabel('best trigger words for each query', fontsize=12)
        plt.ylabel('query index', fontsize=9)
        # plt.yticks(np.arange(0.5, len(topdoc_trigger), 1), [
        #           d['q_id'] for d in all_best_trigger_dict], rotation=0, fontsize=8)
        plt.yticks(np.arange(0.5, len(topdoc_trigger), 1),
                   np.arange(len(topdoc_trigger)), rotation=0, fontsize=8)
        plt.title(
            f'Heatmap of counts of the most frequent\n triggers with {length}_trigger_length', fontsize=14)
        fig_path = plot_path/f'{length}_Heatmap_frequent_count.png'
        plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    plot()
    # draw_heatmap()
