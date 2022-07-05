"""
This file conduct experiments on the effect of attack positions.
Positions includes 'middle', 'end', 'random', 'max_gradient', and 'min_gradient'.
For 'start' position, use attack_length.py.
Both 'Clueweb09' and 'msmarco_p' are available.
Both rank demotion and rank promotion scenarios are optional.
Figure 4 illustrates impact of positions of Clueweb09 on both attack scenarios.
"""
from ranklist import device
from utilities.metrics import relative_ranking_changes
import brittle.attack_methods as attack_methods
from operator import index, itemgetter
from copy import deepcopy
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from brittle.attack_run import init_rerank  # get reranker model
from transformers import BertTokenizer  # get related tokenizer
from pytorch_lightning import seed_everything
from utilities.utility import get_embedding_weight, extracted_grads, add_hooks, get_ranking_scores, get_new_score_replace, get_average_grad, get_accuracy, get_best_candidates_topdoc, get_best_candidates_lastdoc, get_loss_per_candidate


def get_trigger_positions_inplace_maxgradient(rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset):
    """Get adversarial token positions that have the maximum gradient scores in the document"""
    inputs = tokenizer([query], [topdoc], padding=True, truncation=True)
    
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)
    loss = rerank_model.model(input_model).squeeze(-1)  

    global extracted_grads
    extracted_grads = []
    loss.backward()
    grads = extracted_grads[0].cpu()
    sum_grads = torch.sum(grads, dim=0)  
    aveg_grads = torch.sum(sum_grads, dim=1)  
    desc_oder = np.argsort(-aveg_grads).tolist()  
    trigger_position = []

    for index in desc_oder:
        if index < trigger_offset or index == len(desc_oder)-1:
            continue
        trigger_position.append(int(index))
        if len(trigger_position) == trigger_token_length:
            break
    return trigger_position


def get_trigger_positions_inplace_mingradient(rerank_model, topdoc, query, tokenizer, trigger_token_length, trigger_offset):
    """Get adversarial token positions that have the minimum gradient scores in the document"""
    inputs = tokenizer([query], [topdoc], padding=True, truncation=True)

    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)
    loss = rerank_model.model(input_model).squeeze(-1) 

    global extracted_grads
    extracted_grads = []
    loss.backward()
    grads = extracted_grads[0].cpu()
    sum_grads = torch.sum(grads, dim=0) 
    aveg_grads = torch.sum(sum_grads, dim=1) 
    asc_oder = np.argsort(aveg_grads).tolist()
    
    trigger_position = []
    for index in asc_oder:
        if index < trigger_offset or index == len(asc_oder)-1:
            continue
        trigger_position.append(int(index))
        if len(trigger_position) == trigger_token_length:
            break
    
    return trigger_position

def run_model(position='inplace_maxgradient', doc_position='lastdoc', dataset='msmarco_p'):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    trigger_token_length = 5  # how many subword pieces in the trigger
    num_candidates = 40
    seed_everything(random_seed, workers=True)

    # device setting loaded from ranklist
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')

    add_hooks(rerank_model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        rerank_model) 

    project_dir = Path(__file__).parent.absolute()
        
    if dataset == 'clueweb09':
        record_path = project_dir / 'Results' / dataset / \
        'fold_1' / 'rank_bert'
        with open(record_path/'query_rank'/'predictions.json', 'r')as f:
            predictions = json.load(f)  # lisr[str]
    if dataset == 'msmarco_p':
        record_path = project_dir / 'Results' / dataset / \
        'fold_1' / 'rank_bert'
        prediction_path = record_path/'query_rank'
        with open(prediction_path/'predictions.json', 'r')as f:
            predictions = json.load(f)  # lisr[str]

    if doc_position == 'topdoc':
        with open(record_path/'id_pair.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'batch_topdoc.json', 'r')as f:
            query_doc = json.load(f)
    if doc_position == 'lastdoc':
        with open(record_path/'id_pair_lastdoc.json', 'r')as f:
            id_pair = json.load(f)
        with open(record_path/'batch_lastdoc.json', 'r')as f:
            query_doc = json.load(f)

    record = []
    for id_index, pair in enumerate(id_pair):
        try:
            q_id = pair[0]
            doc_index = pair[1]
            query = query_doc['queries'][id_index]
            topdoc = query_doc['batch_docs'][id_index]
            orig_predictions = predictions[id_index]
            # get trigger start position --> for gradient collecting, include[CLS] and [SEP]
            query_token_id = tokenizer.encode(query, add_special_tokens=False)
            trigger_offset = len(query_token_id)+2
            # trigger position includes offset of special tokens and query tokens
            
            if position == 'inplace_maxgradient':
                trigger_positions = get_trigger_positions_inplace_maxgradient(
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
            print('trigger_position:', trigger_positions)

            trigger_6_times = []
            for i in range(6):  # different random restarts of the trigger
                # Firstly use 5 [MASK] as triggers.
                if i == 0:
                    trigger_token_ids = torch.tensor(
                        [103]*trigger_token_length)
                    trigger_words = tokenizer.decode(trigger_token_ids)
                    print(
                        f'\nThe 0th time of initializing triggers: with {trigger_token_length} [MASK].')
                else:  # then 5 times randomly initialized triggers
                    trigger_token_ids = np.random.randint(low=999,
                                                          high=total_vocab_size, size=trigger_token_length)
                    trigger_words = tokenizer.decode(trigger_token_ids)
                    print(
                        f'\nThe {i}th time of initializing triggers: randomly.')
                print("initial trigger words: " + trigger_words)
                # get initial loss for trigger
                rerank_model.model.zero_grad()
                loss = get_new_score_replace(rerank_model, topdoc, query,
                                     trigger_token_ids, tokenizer, trigger_positions, trigger_offset)

                print('initial trigger loss:', loss)
                best_loss = loss  # get the loss of doc_index
                new_predictions = get_ranking_scores(
                    orig_predictions, doc_index, loss)
                _, relative_changes = get_accuracy(
                    doc_index, orig_predictions, new_predictions)
                print("relative changes of doc: ", relative_changes)
                
                for _ in range(10):  # updates of the entire trigger sequence
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
                    loss = get_new_score_replace(rerank_model, topdoc, query,
                                         trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
                    print("current trigger: ", trigger_words)
                    print("current loss:", loss)

                    new_predictions = get_ranking_scores(
                        orig_predictions, doc_index, loss)
                    _, relative_changes = get_accuracy(
                        doc_index, orig_predictions, new_predictions)
                    
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
        except:
            continue

    if position == 'inplace_maxgradient':
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

if __name__ == '__main__':
    run_model(position='inplace_maxgradient',
              doc_position='lastdoc', dataset='clueweb09')
