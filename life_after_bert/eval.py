import torch
from torch.utils.data import DataLoader
import tqdm

from life_after_bert.data import collate_fn
from life_after_bert.utils import get_sentence_prob


def evaluate_encoder(model, tokenizer, eval_dataset, device="cpu", batch_size=16, output_predictions=True, progress_bar=True):
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

    Returns: accuracy, (answers, preds)
        accuracy - TODO
        answers - tensor containing ground truths
        preds - tensor containing model predictions
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

        all_answers = torch.tensor(all_answers)  # TODO: only track if output_predictions?
        all_preds = torch.tensor(all_preds)
        output = ((all_answers.numpy() == all_preds.numpy()).mean(),)
        if output_predictions:
            output += ((all_answers, all_preds),)

        return output
    else:
        raise NotImplementedError


def evaluate_decoder(model, tokenizer, eval_dataset, device="cpu", batch_size=16, output_predictions=True, progress_bar=True):
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

        output = ((all_answers.numpy() == all_preds.numpy()).mean(),)
        if output_predictions:
            output += ((all_answers, all_preds),)

        return output
    else:
        raise NotImplementedError


def evaluate_encoder_decoder(model, eval_dataset, static_decoder_input_ids, device="cpu", batch_size=16, output_predictions=True, progress_bar=True):
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
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
                outputs = model(
                    **batch, decoder_input_ids=static_decoder_input_ids.expand(len(batch["input_ids"]), -1).to(device)
                )

                logits = outputs.logits
                choice_logits = logits[:, 1, :].gather(1, choice_ids.to(device))
                max_inds = torch.argmax(choice_logits, dim=1).cpu()
                all_preds.extend(choice_ids.gather(1, max_inds.unsqueeze(1)).squeeze(1).tolist())

        all_answers = torch.tensor(all_answers)
        all_preds = torch.tensor(all_preds)

        output = ((all_answers.numpy() == all_preds.numpy()).mean(),)
        if output_predictions:
            output += ((all_answers, all_preds),)

        return output
    else:
        raise NotImplementedError
