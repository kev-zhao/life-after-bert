import torch
from torch.utils.data import DataLoader
import tqdm

from life_after_bert.data import collate_fn
from life_after_bert.utils import get_sentence_prob


def evaluate_encoder(model, tokenizer, task, eval_dataset, device, mask_token=None, batch_size=16):
    """
    Evaluates any HuggingFace encoder model on a MLM task
    [Explanation of how evaluation works]

    Args:
        model:
            Any pretrained transformers.AutoModelForMaskedLM()
        tokenizer:
            Tokenizer for model
            As eval_dataset already has tokenized input ids, tokenizer is only used for determining the mask token
            and encoding answer choices.
        task:
            Type of task, among `oLMpics MLM`  # TODO: add more tasks
        eval_dataset:
            life_after_bert.data.MCDataset() containing task data
        device:
            "cuda" or "cpu"  # TODO: cuda:0/cuda:1?
        mask_token:
            Model mask token, if different from default (`[MASK]` for BERT, `<mask>` for RoBERTa).
            Main use case is for models like T5, which do not have a mask_token by default in HuggingFace;
            For T5, `<extra_id_0>` should be specified here.
        batch_size:
            # TODO: distributed

    Returns: Tuple of (answers, preds)
        answers - list containing ground truths
        preds - list containing model predictions
    """

    mask_token = mask_token if mask_token is not None else tokenizer.mask_token  # TODO: same code in data.py
    assert mask_token is not None, "mask_token must be provided if tokenizer.mask_token does not exist"
    MASK_ID = tokenizer.encode(mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]

    if task == "oLMpics MLM":
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        all_answers = []
        all_preds = []

        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            for batch_index in range(len(batch["choice_list"])):
                all_answers.append(batch["choice_list"][batch_index][batch["answer_id"][batch_index]])

            choice_lists = batch.pop("choice_list")
            del batch["answer_id"]
            for key, value in batch.items():
                batch[key] = value.to(device)

            with torch.no_grad():
                outputs = model(**batch)

                logits = outputs.logits

                # TODO: parallelize
                for batch_index, logit in enumerate(logits):  # Assuming all are single tokens
                    choice_ids = torch.tensor([  # .item() will fail if multiple tokens per word
                        tokenizer.encode(
                            " " + choice_lists[batch_index][num_choices], add_special_tokens=False, return_tensors="pt"
                        ).item()
                        for num_choices in range(len(choice_lists[0]))
                    ])

                    MASK_INDEX = batch["input_ids"][batch_index].tolist().index(MASK_ID)
                    probs = logit[MASK_INDEX].index_select(0, choice_ids.to(device))

                    max_ind = torch.argmax(probs)
                    all_preds.append(choice_lists[batch_index][max_ind])

        return all_answers, all_preds
    else:
        raise NotImplementedError


def evaluate_decoder(model, tokenizer, task, num_choices, eval_dataset, device, mask_token=None, batch_size=16):
    mask_token = mask_token if mask_token is not None else tokenizer.mask_token  # TODO: duplicate code
    assert mask_token is not None, "mask_token must be provided if tokenizer.mask_token does not exist"
    MASK_ID = tokenizer.encode(mask_token, add_special_tokens=False)
    EOS_ID = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
    assert len(MASK_ID) == len(EOS_ID) == 1
    MASK_ID, EOS_ID = MASK_ID[0], EOS_ID[0]

    if task == "oLMpics MLM":
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        all_answers = []
        all_preds = []

        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            mask_indices = []
            eos_indices = []
            for single_input_ids in batch["input_ids"]:
                mask_indices.append(single_input_ids.tolist().index(MASK_ID))
                eos_indices.append(single_input_ids.tolist().index(EOS_ID))

            # choice_list is size (batch_size, num_choices)
            for batch_index in range(len(batch["choice_list"])):
                all_answers.append(batch["choice_list"][batch_index][batch["answer_id"][batch_index]])

            choice_lists = batch.pop("choice_list")
            del batch["answer_id"]
            for key, value in batch.items():
                batch[key] = value.to(device)

            choice_probs = []
            for choice_index in range(num_choices):
                for batch_index in range(len(mask_indices)):
                    batch["input_ids"][batch_index][mask_indices[batch_index]] = tokenizer.encode(
                        " " + choice_lists[batch_index][choice_index], add_special_tokens=False,
                        return_tensors="pt").item()

                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                sentence_probs = get_sentence_prob(batch["input_ids"], logits, eos_indices)
                choice_probs.append(sentence_probs)

            choice_probs = torch.stack(choice_probs, dim=-1)
            max_inds = torch.argmax(choice_probs, dim=-1)

            for batch_index, max_ind in enumerate(max_inds):
                all_preds.append(choice_lists[batch_index][max_ind])

        return all_answers, all_preds
    else:
        raise NotImplementedError
