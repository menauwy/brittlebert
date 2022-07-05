"""
This file conducts experiments on effect of adversarial sequence length.
Here we only consider token sequence added in the start of a document for
 both demotion and promotion scenarios.
"""
from ranklist import device
import brittle.attack_methods as attack_methods
import json
from pathlib import Path
import torch
import numpy as np
from brittle.attack_run import init_rerank  # get reranker model
from transformers import BertTokenizer  # get related tokenizer
from pytorch_lightning import seed_everything
from utilities.utility import get_embedding_weight, extracted_grads, add_hooks, get_ranking_scores, get_new_score, get_average_grad_for_docstart, get_accuracy, get_best_candidates_topdoc_docstart, get_best_candidates_lastdoc_docstart

def run_model(trigger_token_length=5, doc_position='topdoc'):
    random_seed = 1
    total_vocab_size = 30522  # total number of subword pieces in BERT
    trigger_token_length = trigger_token_length
    num_candidates = 40
    seed_everything(random_seed, workers=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    rerank_model = init_rerank('clueweb09', 'bert', 'fold_1')

    add_hooks(rerank_model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(
        rerank_model)


    project_dir = Path(__file__).parent.absolute()
    record_path = project_dir / 'Results' / 'clueweb09' / \
        'fold_1' / 'rank_bert'
    prediction_path = record_path/'query_rank'
    with open(prediction_path/'predictions.json', 'r')as f:
        predictions = json.load(f)

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
        q_id = pair[0]
        doc_index = pair[1]
        orig_predictions = predictions[id_index]
        query = query_doc['queries'][id_index]
        topdoc = query_doc['batch_docs'][id_index]
        
        query_token_id = tokenizer.encode(query, add_special_tokens=False)
        trigger_start_position = len(query_token_id)+2
        
        trigger_6_times = []
        for i in range(6):  # different random restarts of the trigger
            # Firstly use 5 [MASK] as triggers.
            if i == 0:
                trigger_token_ids = torch.tensor([103]*trigger_token_length)
                trigger_words = tokenizer.decode(trigger_token_ids)
                print(
                    f'\nThe 0th time of initializing triggers for: with {trigger_token_length} [MASK].')
            else:  # then 5 times randomly initialize triggers
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
            _, relative_changes = get_accuracy(
                doc_index, orig_predictions, new_predictions)
            print("relative changes of doc: ", relative_changes)
            
            for _ in range(20):  # updates of the entire trigger sequence
                averaged_grad = get_average_grad_for_docstart(
                    loss, trigger_token_ids, trigger_start_position)  # shape of (5, 768)

                if doc_position == 'topdoc':
                    cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                        averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=False)
                    trigger_token_ids = get_best_candidates_topdoc_docstart(rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1)
                if doc_position == 'lastdoc':
                    cand_trigger_token_ids = attack_methods.hotflip_attack_remove_unused(
                        averaged_grad, embedding_weight, trigger_token_ids, num_candidates=num_candidates, increase_loss=True)
                    trigger_token_ids = get_best_candidates_lastdoc_docstart(rerank_model, topdoc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1)

                trigger_words = tokenizer.decode(trigger_token_ids)
                loss = get_new_score(rerank_model, topdoc, query,
                                     trigger_token_ids, tokenizer)
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

    with open(record_path/'length_record'/f'{doc_position}_trigger{trigger_token_length}_record.json', 'w')as f:
        json.dump(record, f)

if __name__ == '__main__':
    run_model(trigger_token_length=1, doc_position='topdoc')
    run_model(trigger_token_length=3, doc_position='topdoc')
    run_model(trigger_token_length=5, doc_position='topdoc')
    run_model(trigger_token_length=7, doc_position='topdoc')
    run_model(trigger_token_length=9, doc_position='topdoc')
    run_model(trigger_token_length=13, doc_position='topdoc')
    run_model(trigger_token_length=20, doc_position='topdoc')
