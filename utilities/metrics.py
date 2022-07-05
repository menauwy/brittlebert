""" metrics for attacking models: normalized ranking shift"""

def _get_rankpairs(doc_id: int, predictions: torch.FloatTensor):
    # rankpair tuple as dict key, dict value is a boolean value,
    # denote whether doc_left has higher score than doc_right
    # e.g. {(doc_id1, doc_id2): True} denotes doc_id1 has higher importance score

    doc_score = predictions[doc_id]
    rankpairs = {}
    for index, prediction in enumerate(predictions):
        if index == doc_id:
            continue
        if doc_score >= prediction:
            rankpairs[(doc_id, index)] = True
        elif doc_score < prediction:
            rankpairs[(doc_id, index)] = False
    # delete self compare if exits ??
    if (doc_id, doc_id) in rankpairs:
        del rankpairs[(doc_id, doc_id)]
    return rankpairs


def relative_ranking_changes(doc_id: int, orig_predictions: torch.FloatTensor, new_predictions: torch.FloatTensor):
    """for specific document, calculate relative ranking changes
    between rerank model ranking list and attacked model ranking list"""
    # get all ranking pairs for this doc_id
    orig_rankpairs = _get_rankpairs(doc_id, orig_predictions)
    new_rankpairs = _get_rankpairs(doc_id, new_predictions)
    changes = [[k, v] for k, v in new_rankpairs.items(
    ) if new_rankpairs[k] != orig_rankpairs[k]]
    relative_percentage = len(changes)/(len(orig_predictions)-1)
    # return changed rankpairs after attaction & relative percentage
    return changes, relative_percentage
