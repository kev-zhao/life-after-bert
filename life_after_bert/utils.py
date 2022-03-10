import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sentence_prob(input_ids, logits, eos_indices):
    # TODO: check with Namrata that "https://github.com/NamrataRShivagunde/oLMpics/blob/gpt2-runs/oLMpics/gpt2_mc_mlm.py" is latest version
    # TODO: improve docstring
    """
    Computes the probability for a sentence in a batch
    Arguments:
        input_ids (LongTensor) : Input ids for a batch, size [batch_size, seq_len]
        logits (FloatTensor) : Output logits of size [batch_size, seq_len, vocab_size]
        eos_indices (list) : List of the index of the end token for each question.
    Returns
        probs (FloatTensor) : of size [batch_size]
                              E.g.: If there are two choices for a given question, there will be two sentence
                              probabilities as there would be two times the [MASK] is replaced by the given choices and sentence probability is computed.
    """
    # Multiplies together individual probabilities to get overall sentence probability
    logits = F.softmax(logits, dim=2)
    probs = torch.gather(logits, 2, input_ids.unsqueeze(-1)[:, 1:]).squeeze(-1)  # Shift the logit left by one
    for i in range(len(eos_indices)):  # Set probability of pad tokens at end to 1
        probs[i, eos_indices[i] - 1:] = 1  # TODO: check these are all <pad>s (EOS for GPT?)

    probs = torch.sum(torch.log(probs), dim=1)
    return probs
