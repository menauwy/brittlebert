from ranklist import device
from transformers.utils.dummy_tf_objects import TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST
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


def get_new_score(rerank_model, topdoc, query, trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
    """
    Takes a batch of documents(should only include doc1) for each query, and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    # convert ids to words
    newdoc = deepcopy(topdoc)
    doc_token_ids = tokenizer.encode(newdoc, add_special_tokens=False)
    for i, position in enumerate(trigger_positions):
        doc_token_ids[position-trigger_offset] = trigger_token_ids[i]
    newdoc = tokenizer.decode(doc_token_ids)

    # reference dataIterBert.py _collate_bert
    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    # torch.LongTensor of shape (batch_size, sequence_length)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)  # tensor([2.2663])

    return out


def get_new_score_insert(rerank_model, topdoc, query, trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
    """
    Takes a batch of documents(should only include doc1) for each query, and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    # convert ids to words
    newdoc = deepcopy(topdoc)
    doc_token_ids = tokenizer.encode(newdoc, add_special_tokens=False)
    for i, position in enumerate(trigger_positions):
        front_part = doc_token_ids[:position-trigger_offset]
        end_part = doc_token_ids[position-trigger_offset:]
        doc_token_ids = front_part + trigger_token_ids[i] + end_part
    newdoc = tokenizer.decode(doc_token_ids)

    # reference dataIterBert.py _collate_bert
    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    # torch.LongTensor of shape (batch_size, sequence_length)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)  # tensor([2.2663])

    return out


def get_trigger_positions_inplace(rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset):

    inputs = tokenizer([query], [topdoc], padding=True, truncation=True)
    #print('tokens of the inputs are:',inputs)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)
    loss = rerank_model.model(input_model).squeeze(-1)  # tensor([2.2663])

    global extracted_grads
    extracted_grads = []
    loss.backward()
    grads = extracted_grads[0].cpu()
    sum_grads = torch.sum(grads, dim=0)  # torch.Size([512,768])/[290,768]
    aveg_grads = torch.sum(sum_grads, dim=1)  # torch.Size([512])/[290]
    #print('aveg_grad:', aveg_grads)
    desc_oder = np.argsort(-aveg_grads).tolist()  # descending order
    #print('desc oder of indices: ', desc_oder)
    # inplace should occur after query tokens and special tokens
    #print('desc_oder:', desc_oder)
    trigger_position = []
    for index in desc_oder:
        if index < trigger_offset or index == len(desc_oder)-1:
            continue
        trigger_position.append(int(index))
        if len(trigger_position) == trigger_token_length:
            break
    #print('trigger_postion:', trigger_position)
    return trigger_position


def get_trigger_positions_inplace_mingradient(rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset):

    inputs = tokenizer([query], [topdoc], padding=True, truncation=True)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)
    loss = rerank_model.model(input_model).squeeze(-1)  # tensor([2.2663])

    global extracted_grads
    extracted_grads = []
    loss.backward()
    grads = extracted_grads[0].cpu()
    sum_grads = torch.sum(grads, dim=0)  # torch.Size([520,768])
    aveg_grads = torch.sum(sum_grads, dim=1)  # torch.Size([512])
    #print('aveg_grad:', aveg_grads)
    asc_oder = np.argsort(aveg_grads).tolist()
    # inplace should occur after query tokens and special tokens
    #print('desc_oder:', desc_oder)
    trigger_position = []
    for index in asc_oder:
        if index < trigger_offset or index == len(asc_oder)-1:
            continue
        trigger_position.append(int(index))
        if len(trigger_position) == trigger_token_length:
            break
    #print('trigger_postion:', trigger_position)
    return trigger_position


def get_average_grad(loss, trigger_token_ids, trigger_positions):
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
    grads = extracted_grads[0].cpu()  # 再确定一下维度
    # print(extracted_grads)  #[torch.Size([1,290,768])] list of tensor
    # print(grads)
    # print(grads.shape)  # torch.Size([1,290,768]) tensor of 3 dims [B,V,E]
    # print(grads[0].shape)
    # average grad across batch size, result only make sense for trigger token at the front
    # sum up in batch dim --> torch.Size([290,768])
    averaged_grad = torch.sum(grads, dim=0)
    # print(averaged_grad.shape)#[290,768]
    # return just trigger grads
    res = averaged_grad[trigger_positions[0]].unsqueeze(0)
    for i in range(1, len(trigger_positions)):
        res = torch.cat(
            (res, averaged_grad[trigger_positions[i]].unsqueeze(0)), dim=0)
    return res


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


def get_best_candidates_topdoc(rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1):
    """
    Given the list of candidate trigger token ids(of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
    # print(loss_per_candidate)
    # (array([ 2732, 10799,  9845]), array([1.9463267], dtype=float32))....

    # maximize the loss
    # top_candidates = heapq.nsmallest(
    #    beam_size, loss_per_candidate, key=itemgetter(1))

    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
        :beam_size]

    # top_candidates now contains beam_size trigger sequences, each with a different 0th tokem
    # for all trigger tokens, skipping the 0th (done ablve)
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, topdoc, query, cand, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset))

        # top_candidates = heapq.nsmallest(  # 改 不一定用heapq
        #    beam_size, loss_per_candidate, key=itemgetter(1))
        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
            :beam_size]

        #print('top:', top_candidates)
    return min(top_candidates, key=itemgetter(1))[0]


def get_best_candidates_lastdoc(rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1):
    """
    Given the list of candidate trigger token ids(of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
    # print(loss_per_candidate)
    # (array([ 2732, 10799,  9845]), array([1.9463267], dtype=float32))....

    # maximize the loss
    # top_candidates = heapq.nsmallest(
    #    beam_size, loss_per_candidate, key=itemgetter(1))

    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
        :beam_size]

    # top_candidates now contains beam_size trigger sequences, each with a different 0th tokem
    # for all trigger tokens, skipping the 0th (done ablve)
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, topdoc, query, cand, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset))

        # top_candidates = heapq.nsmallest(  # 改 不一定用heapq
        #    beam_size, loss_per_candidate, key=itemgetter(1))
        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
            :beam_size]

        #print('top:', top_candidates)
    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(index, rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
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
                              trigger_token_ids, tokenizer, trigger_positions, trigger_offset).cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_one_replaced = deepcopy(
            trigger_token_ids)  # copy triggers
        # replace one token
        trigger_token_one_replaced[index] = cand_trigger_token_ids[index][cand_id]
        loss = get_new_score(rerank_model, topdoc, query,
                             trigger_token_one_replaced, tokenizer, trigger_positions, trigger_offset).cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_one_replaced), loss))
    return loss_per_candidate

# def make_target_batch(tokenizer, device, target_texts): 补充


def run_model(position='inplace', doc_position='lastdoc', dataset='msmarco_p'):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    trigger_token_length = 5  # how many subword pieces in the trigger
    num_candidates = 40
    seed_everything(random_seed, workers=True)

    # device 的设置在ranklist
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')
    # 我们只使用bert model，所以这里哪个dataset无所谓。
    # model 已经是.to(device).eval()

    add_hooks(rerank_model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        rerank_model)  # save the word embedding matrix
    # shape: [30522,768]

    # target label can be reloaded according to the doc we used
    # doc_id = 'clueweb09-en0008-49-09140'
    project_dir = Path('/home/wang/attackrank')
    record_path = project_dir / 'Results' / dataset / \
        'fold_1' / 'rank_bert'
    #prediction_path = record_path/'query_rank'
    prediction_path = record_path/'200_query'
    with open(prediction_path/'200_prediction.json', 'r')as f:
        predictions = json.load(f)  # lisr[str]

    if doc_position == 'topdoc':
        with open(record_path/'id_pair.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'batch_topdoc.json', 'r')as f:
            query_doc = json.load(f)
    if doc_position == 'lastdoc':
        """
        with open(record_path/'id_pair_last.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'batch_lastdoc.json', 'r')as f:
            query_doc = json.load(f)
        """
        with open(record_path/'200_query'/'id_pair_lastdoc_200.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'200_query'/'batch_lastdoc_200.json', 'r')as f:
            query_doc = json.load(f)

    record = []
    count = 1
    for id_index, pair in enumerate(id_pair):
        if id_index < 160 or id_index > 199:
            continue
        try:
            q_id = pair[0]
            doc_index = pair[1]
            query = query_doc['queries'][id_index]
            topdoc = query_doc['batch_docs'][id_index]
            orig_predictions = predictions[id_index]
            # get trigger start position --> for gradient collecting
            # include[CLS] and [SEP] [101, 9019, 4391, 102]
            query_token_id = tokenizer.encode(query, add_special_tokens=False)
            trigger_offset = len(query_token_id)+2

            #print('offset:', trigger_offset)
            # get trigger postions: where the gradient is the largest instead in the front of the doc
            # trigger postiion includes offset of special tokens and query tokens
            # list(int)
            if position == 'inplace':
                trigger_positions = get_trigger_positions_inplace(
                    rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset)
            elif position == 'inplace_mingradient':
                trigger_positions = get_trigger_positions_inplace_mingradient(
                    rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset)
            elif position == 'lastplace':
                # get the last postion(511) or the length of topdoc if it <511
                begin = min(511, len(tokenizer.encode(
                    topdoc, add_special_tokens=False))+trigger_offset)
                trigger_positions = list(
                    np.arange(start=begin, stop=begin-trigger_token_length, step=-1))[::-1]
            elif position == 'middleplace':
                begin = min(256, (len(tokenizer.encode(
                    topdoc, add_special_tokens=False))+trigger_offset)//2)
                trigger_positions = list(
                    np.arange(start=begin, stop=begin-trigger_token_length, step=-1))[::-1]
            elif position == 'random':
                last = min(511, len(tokenizer.encode(
                    topdoc, add_special_tokens=False)))
                trigger_positions = sorted(np.random.randint(
                    low=trigger_offset, high=last, size=trigger_token_length).tolist())
            # list[int*trigger_token_length]
            print('trigger_position:', trigger_positions)
            # 每个trigger 加上offset

            #rerank_model._init_query(q_id, rank_scores=True)

            trigger_6_times = []
            for i in range(6):  # different random restarts of the trigger
                # Firstly use 5 [MASK] as triggers.
                if i == 0:
                    trigger_token_ids = torch.tensor(
                        [103]*trigger_token_length)
                    trigger_words = tokenizer.decode(trigger_token_ids)
                    print(
                        f'\nThe 0th time of initializing triggers: with {trigger_token_length} [MASK].')
                else:  # then 5 times randomly initialize triggers
                    # sample random initial trigger ids
                    trigger_token_ids = np.random.randint(low=999,
                                                          high=total_vocab_size, size=trigger_token_length)
                    trigger_words = tokenizer.decode(trigger_token_ids)
                    print(
                        f'\nThe {i}th time of initializing triggers: randomly.')
                print("initial trigger words: " + trigger_words)
                # get initial loss for trigger
                rerank_model.model.zero_grad()
                loss = get_new_score(rerank_model, topdoc, query,
                                     trigger_token_ids, tokenizer, trigger_positions, trigger_offset)

                print('initial trigger loss:', loss)
                best_loss = loss  # get the loss of doc_index
                new_predictions = get_ranking_scores(
                    orig_predictions, doc_index, loss)
                changes, relative_changes = get_accuracy(
                    doc_index, orig_predictions, new_predictions)
                print("relative changes of doc: ", relative_changes)
                # counter = 0
                # end_iter = False
                for _ in range(10):  # this many updates of the entire trigger sequence
                    averaged_grad = get_average_grad(
                        loss, trigger_token_ids, trigger_positions)  # shape of (5, 768)

                    if doc_position == 'topdoc':
                        cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                            averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=False)

                        trigger_token_ids = get_best_candidates_topdoc(
                            rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1)

                    if doc_position == 'lastdoc':
                        cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                            averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=True)

                        trigger_token_ids = get_best_candidates_lastdoc(
                            rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1)

                    trigger_words = tokenizer.decode(trigger_token_ids)
                    loss = get_new_score(rerank_model, topdoc, query,
                                         trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
                    print("current trigger: ", trigger_words)
                    print("current loss:", loss)

                    new_predictions = get_ranking_scores(
                        orig_predictions, doc_index, loss)
                    changes, relative_changes = get_accuracy(
                        doc_index, orig_predictions, new_predictions)
                    # print('orig_prediction:\n', orig_predictions[doc_index])
                    # print('new_prediction:\n', new_predictions[doc_index])
                    # print("Changes of ranking of doc: ", changes)
                    print("relative changes of doc: ", relative_changes)
                    if doc_position == 'topdoc':
                        if loss < best_loss:
                            best_loss = loss
                        else:
                            break
                    if doc_position == 'lastdoc':
                        if loss > best_loss:
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

            # save steps
            """
            if doc_index % 30 == 0:
                if position == 'inplace':
                    with open(record_path/'position_record'/f'inplace_{doc_position}_trigger5_record{count}.json', 'w')as f:
                        json.dump(record, f)
                    count += 1
                    record = []
            """
        except:
            continue

    if position == 'inplace':
        with open(record_path/'position_record'/f'inplace_{doc_position}_trigger5_record_200_5.json', 'w')as f:
            json.dump(record, f)
    elif position == 'inplace_mingradient':
        with open(record_path/'position_record'/f'inplace_mingradient_{doc_position}_trigger5_record.json', 'w')as f:
            json.dump(record, f)
    elif position == 'lastplace':
        with open(record_path/'position_record'/f'lastplace_{doc_position}_trigger5_record.json', 'w')as f:
            json.dump(record, f)
    elif position == 'middleplace':
        with open(record_path/'position_record'/f'middleplace_{doc_position}_trigger5_record.json', 'w')as f:
            json.dump(record, f)
    elif position == 'random':
        with open(record_path/'position_record'/f'random_{doc_position}_trigger5_record.json', 'w')as f:
            json.dump(record, f)


def same_trigger_diff_position():

    def attack_position(position='topplace'):
        random_seed = 1
        total_vocab_size = 30522  # total number of subword pieces in BERT
        trigger_token_length = 5  # how many subword pieces in the trigger
        num_candidates = 40
        seed_everything(random_seed, workers=True)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')

        add_hooks(rerank_model)  # add gradient hooks to embeddings
        embedding_weight = get_embedding_weight(
            rerank_model)

        project_dir = Path('/home/wang/attackrank')
        record_path = project_dir / 'Results' / 'clueweb09' / \
            'fold_1' / 'rank_bert'
        prediction_path = record_path/'query_rank'
        with open(prediction_path/'predictions.json', 'r')as f:
            predictions = json.load(f)
        with open(record_path/'id_pair.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'batch_topdoc.json', 'r')as f:
            query_doc = json.load(f)

        # 1. get global triggers as fixed triggers
        with open(record_path/'batch_topdoc_trigger5_record_without_unused.json', 'r')as f:
            batch_trigger_record = json.load(f)
        trigger_token_ids = batch_trigger_record[0][-1]['trigger_token_ids']
        print('trigger token ids:', trigger_token_ids)

        relative_changes_for_position_attack = []
        for id_index, pair in enumerate(id_pair):
            q_id = pair[0]
            doc_index = pair[1]
            query = query_doc['queries'][id_index]
            topdoc = query_doc['batch_docs'][id_index]
            # get trigger start position --> for gradient collecting
            # include[CLS] and [SEP] [101, 9019, 4391, 102]
            query_token_id = tokenizer.encode(query, add_special_tokens=False)
            trigger_offset = len(query_token_id)+2

            # 2. get trigger positions
            if position == 'topplace':
                trigger_positions = list(
                    np.arange(start=trigger_offset, stop=trigger_offset+trigger_token_length))
            if position == 'inplace_maxgradient':
                trigger_positions = get_trigger_positions_inplace(
                    rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset)
            elif position == 'inplace_mingradient':
                trigger_positions = get_trigger_positions_inplace_mingradient(
                    rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset)
            elif position == 'lastplace':
                # get the last postion(511) or the length of topdoc if it <511
                begin = min(511, len(tokenizer.encode(
                    topdoc, add_special_tokens=False)))
                trigger_positions = list(
                    np.arange(start=begin, stop=begin-trigger_token_length, step=-1))[::-1]
            elif position == 'middleplace':
                begin = min(256, len(tokenizer.encode(
                    topdoc, add_special_tokens=False))//2)
                trigger_positions = list(
                    np.arange(start=begin, stop=begin-trigger_token_length, step=-1))[::-1]
            elif position == 'random':
                last = min(511, len(tokenizer.encode(
                    topdoc, add_special_tokens=False)))
                trigger_positions = sorted(np.random.randint(
                    low=trigger_offset, high=last, size=trigger_token_length).tolist())
            # list[int*trigger_token_length]
            print('trigger_position:', trigger_positions)

            loss = get_new_score(rerank_model, topdoc, query,
                                 trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
            orig_predictions = predictions[id_index]
            new_predictions = get_ranking_scores(
                orig_predictions, doc_index, loss)
            changes, relative_changes = get_accuracy(
                doc_index, orig_predictions, new_predictions)

            relative_changes_for_position_attack.append(relative_changes)
        return relative_changes_for_position_attack
    positions = ['lastplace', 'middleplace',
                 'inplace_mingradient', 'inplace_maxgradient', 'topplace', 'random']
    all_relative_changes = []
    for position in positions:
        relative_changes_for_position_attack = attack_position(
            position=position)
        all_relative_changes.append(relative_changes_for_position_attack)

    # plot in one picture
    plot_path = Path(
        '/home/wang/attackrank/Results/clueweb09/fold_1/rank_bert/position_record')
    plt.figure(figsize=(10, 8))
    colors = ['slateblue', 'seagreen', 'grey',
              'darkviolet', 'indianred', 'darkorange']
    for i in range(len(colors)):
        plt.plot(all_relative_changes[i],
                 color=colors[i], lw=2, label=positions[i])
    plt.xlabel('query index', fontsize=12)
    plt.ylabel('relative changes', fontsize=12)
    plt.title(
        'Attack topdoc with the same trigger sequence but in different positions', fontsize=14)
    plt.legend()
    plt.savefig(plot_path/'same_trigger_diff_positions.png')


if __name__ == '__main__':
    # run_model(position='inplace', doc_position='lastdoc')
    #run_model(position='inplace_mingradient', doc_position='lastdoc')
    #run_model(position='lastplace', doc_position='lastdoc')
    run_model(position='inplace',
              doc_position='lastdoc', dataset='clueweb09')
    #run_model(position='random', doc_position='lastdoc')
    # same_trigger_diff_position()
