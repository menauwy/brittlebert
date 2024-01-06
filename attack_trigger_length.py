from os import dup
from ranklist import device
from utilities.metrics import relative_ranking_changes
import attack_methods
from operator import index, itemgetter
from copy import deepcopy
import json
from pathlib import Path
import sys
import heapq
import torch
import torch.nn.functional as F
import numpy as np
from attack_run import init_rerank  # get reranker model
from transformers import BertTokenizer  # get related tokenizer
from pytorch_lightning import seed_everything
from collections import Counter
import matplotlib.pylab as plt
# sys.path.append('..')


def get_embedding_weight(rerank_model):
    for module in rerank_model.model.bert.modules():
        if isinstance(module, torch.nn.Embedding):
            # Bert has 5 embedding layers, only add a hook to wordpiece embeddings
            print(module.weight.shape[0])
            # BertModel.embeddingsword_embeddings.weight.shape == (30522,768)
            if module.weight.shape[0] == 30522:
                return module.weight.detach()


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])  # grad_out is a tuple(tensor,)

# add hooks for word embeddings


def add_hooks(rerank_model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    for module in rerank_model.model.bert.modules():
        if isinstance(module, torch.nn.Embedding):
            # only add a hook to wordpiece embeddings
            if module.weight.shape[0] == 30522:
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)


def get_ranking_scores(orig_predictions, doc_index, loss):
    """
    #used to generate all ranking scores for a query, further for evaluation
    #多此一举(orig_predictions, doc_index, loss)
    orig_doc = deepcopy(rerank_model.InferenceDataset.top_docs[doc_index])
    rerank_model.InferenceDataset.top_docs[doc_index] = trigger_words + ' ' + orig_doc
    prediction = rerank_model._rank_docs(
        rerank_model.InferenceDataset.query, rerank_model.InferenceDataset.top_docs)

    rerank_model.InferenceDataset.top_docs[doc_index] = orig_doc
    return torch.tensor(prediction)
    """
    new_predictions = deepcopy(orig_predictions)
    new_predictions[doc_index] = loss
    return new_predictions


def get_new_score(rerank_model, topdoc, query, trigger_token_ids, tokenizer):
    """
    Takes a batch of documents(should only include doc1) for each query, and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    # convert ids to words
    trigger_words = tokenizer.decode(trigger_token_ids)
    newdoc = trigger_words + ' ' + topdoc
    # reference dataIterBert.py _collate_bert
    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    # torch.LongTensor of shape (batch_size, sequence_length)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)  # tensor([2.2663])

    return out


def get_average_grad(loss, trigger_token_ids, trigger_start_position):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch(or doc1). No target label, gradient on the model ouput score.
    """
    #optimizer = torch.optim.Adam(rerank_model.model.parameters())
    # optimizer.zero_grad()

    global extracted_grads
    extracted_grads = []  # clear existing stored grads
    # loss = get_loss(rerank_model, doc_index, q_id, trigger_words)
    loss.backward()
    grads = extracted_grads[0].cpu()
    # print(extracted_grads)  #[torch.Size([1,290,768])] list of tensor
    # print(grads)
    # print(grads.shape)  # torch.Size([1,290,768]) tensor of 3 dims [B,V,E]
    # print(grads[0].shape)
    # average grad across batch size, result only make sense for trigger token at the front
    # sum up in batch dim --> torch.Size([290,768])
    averaged_grad = torch.sum(grads, dim=0)
    # print(averaged_grad.shape)#[290,768]
    # return just trigger grads
    l = len(trigger_token_ids)

    averaged_grad = averaged_grad[trigger_start_position:
                                  trigger_start_position+l]
    return averaged_grad


def get_accuracy(doc_index, orig_predictions, new_predictions):
    """
    rerank_model: Rerank, initialized outside the function

    use metrics defined in the metrics.py
    named with relative_ranking_changes.
    used for monitoring output together with loss.
    check ranking changes for doc1 with/without trigger in every itreration

    # step1: rerank without trigger tokens(load from orig_predicitons)
    # 不能用rerank_model._init_query(q_id, rank_scores=True)
    # orig_predictions = rerank_model.InferenceDataset.prediction

    # save original doc without triggers & concatenate trigger to doc
    # trigger_words = tokenizer.convert_ids_to_tokens(trigger_token_ids)
    orig_doc = deepcopy(rerank_model.InferenceDataset.top_docs[doc_index])
    rerank_model.InferenceDataset.top_docs[doc_index] = trigger_words + ' ' + orig_doc

    # Step2: rerank with trigger tokens
    new_predictions = rerank_model._rank_docs(
        rerank_model.InferenceDataset.query, rerank_model.InferenceDataset.top_docs)
    new_predictions = torch.Tensor(new_predictions)
    # change document back
    rerank_model.InferenceDataset.top_docs[doc_index] = orig_doc
    """

    # step3: use metric to get relative changes
    changes, relative_changes = relative_ranking_changes(
        doc_index, orig_predictions, new_predictions)

    return changes, relative_changes


def get_best_candidates(rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1):
    """
    Given the list of candidate trigger token ids(of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer)
    # print(loss_per_candidate)
    # (array([ 2732, 10799,  9845]), array([1.9463267], dtype=float32))....

    # maximize the loss
    # top_candidates = heapq.nsmallest(
    #    beam_size, loss_per_candidate, key=itemgetter(1))

    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
        :beam_size]
    print('top:', top_candidates)
    # top_candidates now contains beam_size trigger sequences, each with a different 0th tokem
    # for all trigger tokens, skipping the 0th (done ablve)
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, topdoc, query, cand, cand_trigger_token_ids, tokenizer))

        # top_candidates = heapq.nsmallest(  # 改 不一定用heapq
        #    beam_size, loss_per_candidate, key=itemgetter(1))
        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
            :beam_size]

        print('top:', top_candidates)
    return min(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(index, rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer):
    """
    for a particular index, the function tries all of the candidate tokens for that index
    The function returns a list containing the candidta triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
        print("only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates: np.array
    curr_loss = get_new_score(rerank_model, topdoc, query,
                              trigger_token_ids, tokenizer).cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_one_replaced = deepcopy(
            trigger_token_ids)  # copy triggers
        # replace one token
        trigger_token_one_replaced[index] = cand_trigger_token_ids[index][cand_id]
        loss = get_new_score(rerank_model, topdoc, query,
                             trigger_token_one_replaced, tokenizer).cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_one_replaced), loss))
    return loss_per_candidate

# def make_target_batch(tokenizer, device, target_texts): 补充


def run_model(trigger_token_length=5):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    trigger_token_length = trigger_token_length
    num_candidates = 40
    seed_everything(random_seed, workers=True)

    # device 的设置在ranklist
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')
    # model 已经是.to(device).eval()

    add_hooks(rerank_model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        rerank_model)  # save the word embedding matrix
    # shape: [30522,768]

    # target label can be reloaded according to the doc we used
    # doc_id = 'clueweb09-en0008-49-09140'
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    prediction_path = record_path/'query_rank'
    with open(record_path/'id_pair.json', 'r')as f:
        id_pair = json.load(f)
    with open(record_path/'batch_topdoc.json', 'r')as f:
        query_doc = json.load(f)

    record = []
    for id_index, pair in enumerate(id_pair):
        q_id = pair[0]
        doc_index = pair[1]
        query = query_doc['queries'][id_index]
        topdoc = query_doc['batch_docs'][id_index]
        # get trigger start position --> for gradient collecting
        # include[CLS] and [SEP] [101, 9019, 4391, 102]
        query_token_id = tokenizer.encode(query, add_special_tokens=False)
        trigger_start_position = len(query_token_id)+2
        #rerank_model._init_query(q_id, rank_scores=True)
        with open(prediction_path/f'{q_id}_prediction.json', 'r')as f:
            orig_predictions = json.load(f)  # lisr[str]

        trigger_6_times = []
        for i in range(6):  # different random restarts of the trigger
            # Firstly use 5 [MASK] as triggers.
            if i == 0:
                trigger_token_ids = torch.tensor([103]*trigger_token_length)
                trigger_words = tokenizer.decode(trigger_token_ids)
                print(
                    f'\nThe 0th time of initializing triggers for: with {trigger_token_length} [MASK].')
            else:  # then 5 times randomly initialize triggers
                # sample random initial trigger ids
                trigger_token_ids = np.random.randint(low=999,
                                                      high=total_vocab_size, size=trigger_token_length)
                trigger_words = tokenizer.decode(trigger_token_ids)
                print(f'\nThe {i}th time of initializing triggers: randomly.')
            print("initial trigger words: " + trigger_words)
            # get initial loss for trigger
            rerank_model.model.zero_grad()
            loss = get_new_score(rerank_model, topdoc, query,
                                 trigger_token_ids, tokenizer)

            print('initial trigger loss:', loss)
            best_loss = loss  # get the loss of doc_index
            new_predictions = get_ranking_scores(
                orig_predictions, doc_index, loss)
            changes, relative_changes = get_accuracy(
                doc_index, orig_predictions, new_predictions)
            print("relative changes of doc: ", relative_changes)
            # counter = 0
            # end_iter = False
            for _ in range(20):  # this many updates of the entire trigger sequence
                """
                for token_to_flip in range(0, trigger_token_length):
                    if end_iter:  # no loss improvement over whole sweep -> continue to new random restart
                        continue
                    """
                # get averaged_grad w.r.t the triggers
                # rerank_model.model.eval()
                averaged_grad = get_average_grad(
                    loss, trigger_token_ids, trigger_start_position)  # shape of (5, 768)

                # print("averaged_grad: ", averaged_grad)
                # use an attack method to get the top candidates
                cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                    averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=False)
                # print("can:", cand_trigger_token_ids)
                # cand_trigger_token_ids = attack_methods.random_attack(embedding_weight,trigger_token_ids,num_candidats)
                # cand_trigger_token_ids = attack_methods.nearst_neighbor_grad(averaged_grad,embedding_weight, trigger_token_ids,tree, 100, decrease_prob=True)
                # query the model to get the best candidates
                trigger_token_ids = get_best_candidates(
                    rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1)
                trigger_words = tokenizer.decode(trigger_token_ids)
                loss = get_new_score(rerank_model, topdoc, query,
                                     trigger_token_ids, tokenizer)
                print("current trigger: ", trigger_words)
                print("current loss:", loss)
                # if the loss didn't get down(means that for No.1 doc,
                # its ranking score didn't go down; for No.100 doc, score didn't go up), break out
                # if best_loss < loss[doc_index] or abs(best_loss - loss[doc_index]) > 1e-5:
                #    break
                # or changes is None
                new_predictions = get_ranking_scores(
                    orig_predictions, doc_index, loss)
                changes, relative_changes = get_accuracy(
                    doc_index, orig_predictions, new_predictions)
                # print('orig_prediction:\n', orig_predictions[doc_index])
                # print('new_prediction:\n', new_predictions[doc_index])
                # print("Changes of ranking of doc: ", changes)
                print("relative changes of doc: ", relative_changes)
                if loss < best_loss:
                    best_loss = loss
                else:
                    break
            print("final trigger words:" + trigger_words)
            trigger_6_times.append({"trigger_words": trigger_words,
                                    "trigger_token_ids": trigger_token_ids.tolist(), "relative_changes": relative_changes})
        print("query_id: ", q_id, "\ndoc_index: ", doc_index)
        print("triggers for 6 times random start:\n", trigger_6_times)
        loss_mean = [dic['relative_changes'] for dic in trigger_6_times]
        print('trigger loss mean:', np.mean(loss_mean))

        exp_dict = {'q_id': q_id,
                    'doc_index': doc_index,
                    'trigger_6_times': trigger_6_times,
                    'trigger_loss_mean': loss_mean}
        record.append(exp_dict)

    with open(record_path/'length_record'/f'topdoc_trigger{trigger_token_length}_record.json', 'w')as f:
        json.dump(record, f)


def get_redundeny():
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    prediction_path = record_path/'query_rank'
    with open(record_path/'id_pair.json', 'r')as f:
        id_pair = json.load(f)
    with open(record_path/'batch_topdoc.json', 'r')as f:
        query_doc = json.load(f)
    with open(record_path/'topdoc_trigger5_bestset.json', 'r')as f:
        best_trigger = json.load(f)

    # initial
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # get top 100 doc of every query
    redun_record = []
    for id_index, pair in enumerate(id_pair):
        redun_dic = {}

        q_id = pair[0]
        doc_index = pair[1]
        query = query_doc['queries'][id_index]
        topdoc = query_doc['batch_docs'][id_index]
        trigger_token_ids = best_trigger[id_index]['trigger_token_ids']

        rerank_model.InferenceDataset.__init_q_docs__(q_id, query)
        top_docs_id = rerank_model.InferenceDataset.top_docs_id
        top_docs = rerank_model.InferenceDataset.top_docs

        top_docs_512 = [tokenizer.encode(doc, add_special_tokens=False)[
            :512] for doc in top_docs]
        top_docs_512_str = [tokenizer.decode(doc) for doc in top_docs_512]
        # doc name are all different, while docs can be the same

        redun_dic['q_id'] = q_id
        redun_dic['doc_num'] = len(top_docs)
        redun_dic['doc_unique_num'] = len(set(top_docs_512_str))

        with open(prediction_path/f'{q_id}_rank.json', 'r')as f:
            orig_ranks = json.load(f)
        with open(prediction_path/f'{q_id}_prediction.json', 'r')as f:
            orig_predictions = json.load(f)
        # orig ranking docs
        # for index in orig_ranks:
        #    print(top_docs_512_str[index][:200])

        # new ranking docs
        rerank_model.model.zero_grad()
        loss = get_new_score(rerank_model, topdoc, query,
                             trigger_token_ids, tokenizer)
        new_predictions = get_ranking_scores(
            orig_predictions, doc_index, loss)
        # changes, relative_changes = get_accuracy(
        #            doc_index, orig_predictions, new_predictions)
        new_rank = np.argsort(-np.array(torch.Tensor(
            new_predictions).detach().cpu().numpy())).tolist()
        # doc index that move up
        acsd_index = new_rank[:new_rank.index(doc_index)]
        print('changed index in ranking order:', acsd_index)
        redun_dic['changed_index_descorder'] = acsd_index
        # doc index that stay in the last
        still_index = new_rank[new_rank.index(doc_index)+1:]
        print('unchanged index in ranking order:', still_index)
        redun_dic['unchanged_index_descorder'] = still_index

        # for i in still_index:
        #    print(top_docs_512_str[i][:200])
        # check desc_doc number & unique number
        still_doc = [top_docs_512_str[i] for i in still_index]
        redun_dic['unchanged_num'] = len(still_doc)
        redun_dic['unique_unchanged_num'] = len(set(still_doc))

        # check wikipedia doc num
        wiki_index = []
        for i, doc in enumerate(top_docs_512_str):
            if 'wikipedia' in doc:
                wiki_index.append(i)
        print('wiki_index:', wiki_index)
        redun_dic['wiki_index'] = wiki_index
        redun_dic['wiki_num'] = len(wiki_index)

        # get wiki doc ranking position
        wiki_rank_position = []
        for i in wiki_index:
            if i in acsd_index:
                wiki_rank_position.append(acsd_index.index(i))
        redun_dic['wiki_rank_position'] = sorted(wiki_rank_position)

        # check how many wiki are still in the end of the rank
        still_wiki_num = 0
        for doc in still_doc:
            if 'wikipedia' in doc:
                still_wiki_num += 1
        redun_dic['unchanged_wiki_num'] = still_wiki_num

        # find duplicates index and print duplicates
        duplicates = [str for str, count in Counter(
            top_docs_512_str).items() if count > 1]
        #print('duplicates:', duplicates)

        dup_list = []
        for string in duplicates:
            single_list = []
            for index, str in enumerate(top_docs_512_str):
                if str == string:
                    single_list.append(index)
            dup_list.append(single_list)
        #print('dup_list:', dup_list)
        redun_dic['duplicates_index_list'] = dup_list

        # get duplicates doc num；
        duplicates_num = 0
        for dup_l in dup_list:
            duplicates_num += len(dup_l) - 1
        redun_dic['duplicates_num'] = duplicates_num

        # get changed duplicates doc num
        changed_dup_num = 0
        for dup_l in dup_list:
            if dup_l[0] not in still_index:
                changed_dup_num += len(dup_l) - 1
        redun_dic['changed_dup_num'] = changed_dup_num

        # get duplicates wiki doc
        dup_wiki_list = []
        dup_wiki_num = 0
        for dup_l in dup_list:
            # because in a dup_l all docs are identical
            if 'wikipedia' in top_docs_512_str[dup_l[0]]:
                dup_wiki_list.append(dup_l)
                dup_wiki_num += len(dup_l) - 1
        redun_dic['duplicates_wiki_list'] = dup_wiki_list
        redun_dic['duplicates_wiki_num'] = dup_wiki_num

        # get changed dup wiki doc
        changed_dup_wiki_num = 0
        for dup_l in dup_wiki_list:
            # because all doc in dup_l get the same score, they change together or not
            if dup_l[0] not in still_index:
                changed_dup_wiki_num += len(dup_l) - 1
        redun_dic['changed_dup_wiki_num'] = changed_dup_wiki_num

        print(redun_dic)
        redun_record.append(redun_dic)

    with open(record_path/'redundency'/'redundency_topdoc_record.json', 'w')as f:
        json.dump(redun_record, f)


def plot_redundency():
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    plot_path = record_path / 'redundency'
    with open(record_path/'id_pair.json', 'r')as f:
        id_pair = json.load(f)
    with open(record_path/'batch_topdoc.json', 'r')as f:
        query_doc = json.load(f)
    with open(record_path/'topdoc_trigger5_bestset.json', 'r')as f:
        best_trigger = json.load(f)
    with open(record_path/'redundency'/'redundency_topdoc_record.json', 'r')as f:
        redund_record = json.load(f)

    # step1: plot line chart with relative best changes
    relative_best_changes = [d['relative_changes'] for d in best_trigger]
    changed_wiki_percentage = [
        (d['wiki_num']-d['unchanged_wiki_num'])/d['doc_num'] for d in redund_record]
    changed_dup_percentage = [d['changed_dup_num'] /
                              d['doc_num'] for d in redund_record]
    changed_dup_wiki_percentage = [
        d['changed_dup_wiki_num']/d['doc_num'] for d in redund_record]

    plt.figure(figsize=(14, 8))
    plt.plot(relative_best_changes, color='#999999', lw=2.5,
             label='with local adversarial tokens')
    plt.plot(changed_wiki_percentage, color='indianred',
             lw=2.5, label='the part of Wikipedia docs')
    plt.plot(changed_dup_percentage, color='slateblue',
             lw=2.5, label='the part of duplicate docs')
    plt.plot(changed_dup_wiki_percentage, color='darkorange',
             lw=2, label='the part of duplicate Wiki docs')
    plt.axvline(x=21, ls='--', lw=2)
    plt.axvline(x=34, ls='--', lw=2)
    plt.axvline(x=36, ls='--', lw=2)
    plt.xlabel('query index', fontsize=16)
    plt.ylabel('normalized rank shift', fontsize=16)
    #plt.title('Document Redundency Analysis', fontsize=16)
    plt.legend(loc='center left', fontsize=14)  # loc='center left'
    plt.savefig(plot_path/'redundency_topdoc_line.png', bbox_inches='tight')
    plt.savefig(plot_path/'redundency_topdoc_line.eps', bbox_inches='tight')
    plt.close()

    # step 2:plot bar chart of duplicates
    duplicates_num = [d['duplicates_num'] for d in redund_record]
    changed_dup_num = [d['changed_dup_num'] for d in redund_record]
    dup_wiki_num = [d['duplicates_wiki_num'] for d in redund_record]
    changed_dup_wiki_num = [d['changed_dup_wiki_num'] for d in redund_record]

    plt.figure(figsize=(12, 4))
    colors = plt.cm.BuPu(np.linspace(0.2, 1.0, 4))[::-1]
    plt.plot(duplicates_num, color=colors[0], lw=2, label='duplicate docs')
    plt.plot(changed_dup_num,
             color=colors[1], lw=2, label='duplicate docs that promote ranks')
    plt.plot(dup_wiki_num, color=colors[2],
             lw=2, label='duplicate Wikipedia docs')
    plt.plot(changed_dup_wiki_num,
             color=colors[3], lw=2, label='duplicate Wiki docs that promote ranks')
    plt.axvline(x=21, ls='--', lw=2)
    plt.axvline(x=34, ls='--', lw=2)
    plt.axvline(x=36, ls='--', lw=2)
    plt.xlabel('query index', fontsize=12)
    plt.ylabel('document number', fontsize=12)
    plt.legend()
    # plt.title(
    #    'Duplicates document ranking changes before & after trigger attack', fontsize=16)
    plt.savefig(plot_path/'duplicates_topdoc_num.png')
    plt.close()

    # step3:plot bar chart of wiki
    wiki_num = [d['wiki_num'] for d in redund_record]
    changed_wiki_num = [d['wiki_num']-d['unchanged_wiki_num']
                        for d in redund_record]
    dup_wiki_num = [d['duplicates_wiki_num'] for d in redund_record]
    changed_dup_wiki_num = [d['changed_dup_wiki_num'] for d in redund_record]
    data = [wiki_num, changed_wiki_num, dup_wiki_num, changed_dup_wiki_num]
    labels = ['Wikipedia docs', 'Wiki docs that promote ranks',
              'duplicate Wiki docs', 'duplicate Wiki docs that promote ranks']

    plt.figure(figsize=(12, 4))
    colors = plt.cm.YlOrBr(np.linspace(0.3, 1.0, 4))[::-1]
    for i, row in enumerate(data):
        plt.plot(row, color=colors[i], lw=2, label=labels[i])
    plt.axvline(x=21, ls='--', lw=2)
    plt.axvline(x=34, ls='--', lw=2)
    plt.axvline(x=36, ls='--', lw=2)
    plt.xlabel('query index', fontsize=12)
    plt.ylabel('document number', fontsize=12)
    plt.legend()
    # plt.title(
    #   'Wikipedia documents ranking changes before & after trriger attack', fontsize=16)
    plt.savefig(plot_path/'wiki_topdoc.png')
    plt.close()


if __name__ == '__main__':
    run_model(trigger_token_length=5)
    # run_model(trigger_token_length=1)
    # run_model(trigger_token_length=3)
    # run_model(trigger_token_length=7)
    # run_model(trigger_token_length=9)
    # run_model(trigger_token_length=13)
    # run_model(trigger_token_length=20)

    # get_redundeny()
    # plot_redundency()
