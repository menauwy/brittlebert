"""
Contains Hotflip method for attacking models. 
Given the gradients for token embeddings, it computes candidates for the final adversarial tokens. 
This code runs on CPU.
"""
import torch
import numpy

def hotflip_attack_remove_unused(averaged_grad, embedding_matrix, trigger_token_ids,
                                 increase_loss=False, num_candidates=1):
    """
    This code is inspired by the code of Paul Michel https://github.com/pmichel31415/translate/blob/
    paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
    and the code of Eric Wallace https://github.com/Eric-Wallace/universal-triggers/blob/master/attacks.py

    Given the model's average_grad over a batch of query-(ir)relevant document pair, the model's whole vacabulary
    token embedding matrix, and the current adversarial token IDs， it returns the top token candidates for each position.

    Note that here we remove all functional tokens including [unused*] tokens from the vocabulary.

    In rank deomotion tasks, where we want to deomote a relevant document, increase_loss is set to False.
    Then the top-k sign-reversed gradients are chosen to decrease the loss, so that the relevance score of the adversarial document
    can be minimized..
    To the contrary, in rank promotion tasks, we set increase_loss as True to increase the loss.
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()

    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))
    # size: [1,trigger_length,30522]

    if not increase_loss:
        # to decrease the loss
        gradient_dot_embedding_matrix *= -1

    # remove all functional tokens including [unused*] tokens from 30522 vocabulary. index:0-998
    removed = gradient_dot_embedding_matrix.numpy()
    removed = numpy.delete(removed, numpy.s_[:999], axis=2)

    removed = torch.from_numpy(removed)

    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(
            removed, num_candidates, dim=2)

        # convert index back to match word embedding version(30522)
        for i, row in enumerate(best_k_ids[0]):
            for j, col in enumerate(row):
                best_k_ids[0][i][j] = col + 999
        #print('best_k_ids:', best_k_ids)
        return best_k_ids.detach().cpu().numpy()[0]

    _, best_at_each_step = removed.max(2)  # dim=2
    for i, row in enumerate(best_at_each_step[0]):
        for j, col in enumerate(row):
            best_at_each_step[0][i][j] = col + 999

    return best_at_each_step[0].detach().cpu().numpy()
