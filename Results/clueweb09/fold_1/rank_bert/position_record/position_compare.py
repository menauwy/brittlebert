from pathlib import Path
import matplotlib.pylab as plt
import json
from collections import Counter
from transformers import BertTokenizer
import numpy as np
import seaborn as sns

project_dir = Path('/home/wang/attackrank')
record_path = project_dir / 'Results' / 'clueweb09' / \
    'fold_1' / 'rank_bert' / 'position_record'
with open(record_path/'inplace_topdoc_trigger5_record.json', 'r')as f:
    inplace = json.load(f)
with open(record_path/'inplace_mingradient_topdoc_trigger5_record.json', 'r')as f:
    inplace_min = json.load(f)
with open(record_path/'lastplace_topdoc_trigger5_record.json', 'r')as f:
    lastplace = json.load(f)
with open(record_path/'middleplace_topdoc_trigger5_record.json', 'r')as f:
    middleplace = json.load(f)
with open(record_path/'topplace_topdoc_trigger5_record.json', 'r')as f:
    topplace = json.load(f)
with open(record_path/'random_topdoc_trigger5_record.json', 'r')as f:
    random = json.load(f)


def plot():
    # get all best relative changes into list[list] shape(5*40)
    all_changes = []
    for record in (lastplace, middleplace, inplace_min, inplace, topplace, random):
        best_changes = []
        for i, res in enumerate(record):
            trigger_6_times = res['trigger_6_times']
            best_trigger = sorted(
                trigger_6_times, key=lambda d: d['relative_changes'])[-1]
            best_changes.append(best_trigger['relative_changes'])
        all_changes.append(best_changes)

    plt.figure(figsize=(10, 6))
    colors = ['slateblue', 'seagreen', 'grey',
              'darkviolet', 'indianred', 'darkorange']
    labels = ['lastplace', 'middleplace',
              'inplace_mingradient', 'inplace_maxgradient', 'topplace', 'random']
    for i in range(len(colors)):
        plt.plot(all_changes[i], color=colors[i],
                 lw=2, label=labels[i])
    plt.xlabel('query index', fontsize=12)
    plt.ylabel('relative changes', fontsize=12)
    plt.title(
        'Comparison of different trigger places on topdoc with 5 triggers', fontsize=14)
    plt.legend()
    plt.savefig(record_path/'comparison_trigger_position.png')


def draw_heatmap():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # get file list in the folder
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    plot_path = record_path / 'position_record'
    file_path = plot_path

    # find path for json file
    positions = ['inplace', 'inplace_mingradient', 'lastplace',
                 'middleplace', 'topplace', 'random']
    for position in positions:
        file = file_path / f'{position}_topdoc_trigger5_record.json'
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
            f'Counts of most frequent trigger words for {position}_attack', fontsize=14)
        for a, b in zip(x, y):
            plt.text(a, b+0.1, b, ha='center', va='bottom')
        plt.savefig(plot_path/f'{position}_Count_frequent_trigger.png')
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

        plt.figure(figsize=(6, 8))
        sns.heatmap(matrix_color, cmap='YlGnBu', cbar_kws={"orientation": "horizontal"},
                    annot=matrix_words, fmt='', annot_kws={'fontsize': 9})
        plt.xlabel('best trigger words for each query', fontsize=12)
        plt.ylabel('query index', fontsize=9)
        # plt.yticks(np.arange(0.5, len(topdoc_trigger), 1), [
        #           d['q_id'] for d in all_best_trigger_dict], rotation=0, fontsize=8)
        plt.yticks(np.arange(0.5, len(topdoc_trigger), 1),
                   np.arange(len(topdoc_trigger)), rotation=0, fontsize=8)
        plt.title(
            f'Heatmap of counts of the most frequent triggers\n for {position}_attack', fontsize=14)
        fig_path = plot_path/f'{position}_Heatmap_frequent_count.png'
        plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    plot()
    # draw_heatmap()
