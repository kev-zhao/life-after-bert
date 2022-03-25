import torch
from torch.utils.data import DataLoader
import tqdm

from life_after_bert.data import collate_fn
from life_after_bert.utils import get_sentence_prob


def evaluate_encoder(model, tokenizer, eval_dataset, device="cpu", batch_size=16, progress_bar=True):
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
        progress_bar:
            Whether or not to use tqdm progress bar

    Returns: Tuple of (answers, preds)
        answers - list containing ground truths
        preds - list containing model predictions
    """

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    mask_id = tokenizer.mask_token_id
    all_answers, all_preds = [], []

    model.to(device)
    model.eval()
    if eval_dataset.task_type == "oLMpics MLM":
        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", disable=not progress_bar):
            all_answers.extend(batch["choice_ids"].gather(1, batch["answer_id"].unsqueeze(1)).squeeze(1).tolist())
            choice_ids = batch.pop("choice_ids")
            del batch["answer_id"]

            for key, value in batch.items():
                batch[key] = value.to(device)

            with torch.no_grad():
                outputs = model(**batch)

                logits = outputs.logits

                mask_indices = torch.tensor([input_ids.tolist().index(mask_id) for input_ids in batch["input_ids"]], device=device)
                mask_logits = logits.gather(1, mask_indices.view(-1, 1, 1).expand(-1, 1, logits.size()[-1])).squeeze(1)
                choice_logits = mask_logits.gather(1, choice_ids.to(device))
                max_inds = torch.argmax(choice_logits, dim=1).cpu()
                all_preds.extend(choice_ids.gather(1, max_inds.unsqueeze(1)).squeeze(1).tolist())

        all_answers = torch.tensor(all_answers)
        all_preds = torch.tensor(all_preds)
        return all_answers, all_preds
    else:
        raise NotImplementedError


def evaluate_decoder(model, tokenizer, eval_dataset, device="cpu", batch_size=16, progress_bar=True):
    """
    TODO
    :param model:
    :param tokenizer:
    :param eval_dataset:
    :param device:
    :param batch_size:
    :param progress_bar:
    :return:
    """

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    all_answers, all_preds = [], []

    model.to(device)
    model.eval()
    if eval_dataset.task_type == "oLMpics MLM":
        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", disable=not progress_bar):
            mask_indices = [input_ids.tolist().index(tokenizer.mask_token_id) for input_ids in batch["input_ids"]]
            eos_indices = [input_ids.tolist().index(tokenizer.eos_token_id) for input_ids in batch["input_ids"]]

            all_answers.extend(batch["choice_ids"].gather(1, batch["answer_id"].unsqueeze(1)).squeeze(1).tolist())
            choice_ids = batch.pop("choice_ids")
            del batch["answer_id"]

            for key, value in batch.items():
                batch[key] = value.to(device)

            choice_probs = []
            for choice_index in range(eval_dataset.num_choices):
                for batch_index in range(len(mask_indices)):  # TODO: remove loop
                    batch["input_ids"][batch_index][mask_indices[batch_index]] = choice_ids[batch_index][choice_index]

                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                sentence_probs = get_sentence_prob(batch["input_ids"], logits, eos_indices)
                choice_probs.append(sentence_probs)

            choice_probs = torch.stack(choice_probs, dim=1)
            max_inds = torch.argmax(choice_probs, dim=1).cpu()

            all_preds.extend(choice_ids.gather(1, max_inds.unsqueeze(1)).squeeze(1).tolist())

        all_answers = torch.tensor(all_answers)
        all_preds = torch.tensor(all_preds)

        return all_answers, all_preds
    else:
        raise NotImplementedError


def evaluate_encoder_decoder(model, tokenizer, task, eval_dataset, static_decoder_input_ids, device,
                             mask_token=None, batch_size=16, progress_bar=True, filter_multi_token_choices=False):
    mask_token = mask_token if mask_token is not None else tokenizer.mask_token  # TODO: same code in data.py
    assert mask_token is not None, "mask_token must be provided if tokenizer.mask_token does not exist"
    MASK_ID = tokenizer.encode(mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]

    if task == "oLMpics MLM":
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        all_answers = []
        all_preds = []

        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", disable=not progress_bar):
            model.eval()

            for batch_index in range(len(batch["choice_list"])):
                all_answers.append(batch["choice_list"][batch_index][batch["answer_id"][batch_index]])

            choice_lists = batch.pop("choice_list")
            del batch["answer_id"]
            for key, value in batch.items():
                batch[key] = value.to(device)

            with torch.no_grad():
                outputs = model(
                    **batch, decoder_input_ids=static_decoder_input_ids.repeat(len(batch["input_ids"]), 1).to(device)
                )

                logits = outputs.logits

                # TODO: parallelize
                all_single_tokens = [True] * len(logits)
                for batch_index, logit in enumerate(logits):  # Assuming all are single tokens
                    try:
                        choice_ids = torch.tensor([  # .item() will fail if multiple tokens per word
                            tokenizer.encode(
                                " " + choice_lists[batch_index][choice_index], add_special_tokens=False, return_tensors="pt"
                            ).item()
                            for choice_index in range(len(choice_lists[0]))
                        ])
                    except ValueError:
                        if filter_multi_token_choices:
                            all_single_tokens[batch_index] = False
                            # break

                        choice_ids = torch.tensor([
                            tokenizer.encode(
                                " " + choice_lists[batch_index][choice_index], add_special_tokens=False,
                                return_tensors="pt"
                            ).squeeze(0)[0]
                            for choice_index in range(len(choice_lists[0]))
                        ])

                        for num_choices in range(len(choice_lists[0])):
                            if len(tokenizer.encode(" " + choice_lists[batch_index][num_choices], add_special_tokens=False)) > 1:
                                print(f"Answer choice more than 1 token: {choice_lists[batch_index][num_choices]}")  # TODO: logger

                    probs = logit[1].index_select(0, choice_ids.to(device))
                    max_ind = torch.argmax(probs)
                    all_preds.append(choice_lists[batch_index][max_ind])

        return all_answers, all_preds
    else:
        raise NotImplementedError
