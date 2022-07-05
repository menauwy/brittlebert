# 
from operator import index, itemgetter
from brittle.ranklist import device
from utilities.metrics import relative_ranking_changes
from operator import index, itemgetter
from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np


def get_embedding_weight(rerank_model):
    for module in rerank_model.model.bert.modules():
        if isinstance(module, torch.nn.Embedding):
            # Bert has 5 embedding layers, only add a hook to wordpiece embeddings
            # BertModel.embeddingsword_embeddings.weight.shape == (30522,768)
            if module.weight.shape[0] == 30522:
                return module.weight.detach()


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0]) 


def add_hooks(rerank_model):
    """
    Add  a hook on the token embedding matrix.
    The gradients w.r.t. the token embeddings will be obtained when loss.backward() 
    is called.
    """
    for module in rerank_model.model.bert.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522:
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)


def get_ranking_scores(orig_predictions, doc_index, loss):
    """
    Replace old prediction with new relevance score.
    """
    new_predictions = deepcopy(orig_predictions)
    new_predictions[doc_index] = loss
    return new_predictions

def get_new_score(rerank_model, doc, query, trigger_token_ids, tokenizer):
    """
    Add adversarial tokens into start positions.
    Note that BERT model has a limited 512 input 
    token entry. We prune the part that is outside this range.
    """
    # convert ids to words
    trigger_words = tokenizer.decode(trigger_token_ids)
    newdoc = trigger_words + ' ' + doc
    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    # torch.LongTensor of shape (batch_size, sequence_length)
    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)

    return out

def get_new_score_replace(rerank_model, doc, query, trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
    """
    Replace adversarial tokens from specific positions.
    Note that BERT model has a limited 512 input 
    token entry. We prune the part that is outside this range.
    """
    # convert ids to words
    newdoc = deepcopy(doc)
    doc_token_ids = tokenizer.encode(newdoc, add_special_tokens=False)
    for i, position in enumerate(trigger_positions):
        doc_token_ids[position-trigger_offset] = trigger_token_ids[i]
    newdoc = tokenizer.decode(doc_token_ids)

    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)  
    return out


def get_new_score_insert(rerank_model, doc, query, trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
    """
    Insert adversarial tokens into specific positions.
    Note that BERT model has a limited 512 input 
    token entry. We prune the part that is outside this range.
    """
    # convert ids to words
    newdoc = deepcopy(doc)
    doc_token_ids = tokenizer.encode(newdoc, add_special_tokens=False)
    for i, position in enumerate(trigger_positions):
        front_part = doc_token_ids[:position-trigger_offset]
        end_part = doc_token_ids[position-trigger_offset:]
        doc_token_ids = front_part + trigger_token_ids[i] + end_part
    newdoc = tokenizer.decode(doc_token_ids)

    inputs = tokenizer([query], [newdoc], padding=True, truncation=True)

    input_ids = torch.LongTensor(inputs['input_ids']).to(device)
    attention_mask = torch.LongTensor(inputs['attention_mask']).to(device)
    token_type_ids = torch.LongTensor(inputs['token_type_ids']).to(device)
    input_model = (input_ids, attention_mask, token_type_ids)

    out = rerank_model.model(input_model).squeeze(-1)  

    return out

def get_average_grad_for_docstart(loss, trigger_token_ids, trigger_start_position):
    """
    Computes the average gradient w.r.t. the adversarial tokens when added in
    a document. No target label, we compute gradients on the model ouput score.
    """
    global extracted_grads
    extracted_grads = [] 
    loss.backward()
    grads = extracted_grads[0].cpu() 
    averaged_grad = torch.sum(grads, dim=0)
    l = len(trigger_token_ids)

    averaged_grad = averaged_grad[trigger_start_position:
                                  trigger_start_position+l]
    return averaged_grad

def get_average_grad(loss, trigger_token_ids, trigger_positions):
    """
    Computes the average gradient w.r.t. the adversarial tokens when added in
    a document. No target label, we compute gradients on the model ouput score.
    """
    global extracted_grads
    extracted_grads = []  
    loss.backward()
    grads = extracted_grads[0].cpu()  
    averaged_grad = torch.sum(grads, dim=0)
    
    res = averaged_grad[trigger_positions[0]].unsqueeze(0)
    for i in range(1, len(trigger_positions)):
        res = torch.cat(
            (res, averaged_grad[trigger_positions[i]].unsqueeze(0)), dim=0)
    return res

def get_accuracy(doc_index, orig_predictions, new_predictions):
    """
    Compute relative ranking shift
    """
    changes, relative_changes = relative_ranking_changes(
        doc_index, orig_predictions, new_predictions)

    return changes, relative_changes


# calculate adversarial token candidates with token positions (inclusive offfset for queries)
def get_best_candidates_topdoc(rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1):
    """
    Given the list of candidate token ids for each position, it uses beam search to find the best new candidate tokens 
    that maximize or minimize the relevance score of a document in line with the task.
    """
    # collects loss for candidates in index 0
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
    
    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
        :beam_size]

    # beam search for candiates from the rest indices.
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, doc, query, cand, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset))

        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
            :beam_size]

    return min(top_candidates, key=itemgetter(1))[0]


def get_best_candidates_lastdoc(rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset, beam_size=1):
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset)
    
    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
        :beam_size]

    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, doc, query, cand, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset))

        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
            :beam_size]

    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(index, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, trigger_positions, trigger_offset):
    """
    Given a specific index, returns all the losses of candidates for that index in a list.
    """
    if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
        return trigger_token_ids
    loss_per_candidate = []
   
    curr_loss = get_new_score_replace(rerank_model, doc, query,
                              trigger_token_ids, tokenizer, trigger_positions, trigger_offset).cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_one_replaced = deepcopy(
            trigger_token_ids) 
        # replace one token
        trigger_token_one_replaced[index] = cand_trigger_token_ids[index][cand_id]
        loss = get_new_score_replace(rerank_model, doc, query,
                             trigger_token_one_replaced, tokenizer, trigger_positions, trigger_offset).cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_one_replaced), loss))
    return loss_per_candidate

# calculate adversarial token candidates for the start of the document
def get_best_candidates_topdoc_docstart(rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1):
    """
    Given the list of candidate token ids for each position, it uses beam search to find the best new candidate tokens 
    that maximize or minimize the relevance score of a document in line with the task.
    """
    # collects loss for candidates in index 0
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer)

    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
        :beam_size]
    # beam search for candiates from the rest indices.
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, doc, query, cand, cand_trigger_token_ids, tokenizer))

        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=False)[
            :beam_size]

    return min(top_candidates, key=itemgetter(1))[0]


def get_best_candidates_lastdoc_docstart(rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer, beam_size=1):
    loss_per_candidate = get_loss_per_candidate(
        0, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer)

    top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
        :beam_size]
    
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:
            loss_per_candidate.extend(get_loss_per_candidate(
                idx, rerank_model, doc, query, cand, cand_trigger_token_ids, tokenizer))

        top_candidates = sorted(loss_per_candidate, key=lambda x: x[1], reverse=True)[
            :beam_size]

    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate_docstart(index, rerank_model, doc, query, trigger_token_ids, cand_trigger_token_ids, tokenizer):
    """
    Given a specific index, returns all the losses of candidates for that index in a list.
    """
    if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
        return trigger_token_ids
    loss_per_candidate = []

    curr_loss = get_new_score(rerank_model, doc, query,
                              trigger_token_ids, tokenizer).cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_one_replaced = deepcopy(
            trigger_token_ids)  
        trigger_token_one_replaced[index] = cand_trigger_token_ids[index][cand_id]
        loss = get_new_score(rerank_model, doc, query,
                             trigger_token_one_replaced, tokenizer).cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_one_replaced), loss))
    return loss_per_candidate