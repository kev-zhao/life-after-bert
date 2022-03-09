import torch
from torch.utils.data import DataLoader
import tqdm


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
    MASK_ID = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]

    if task == "oLMpics MLM":
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        all_answers = []
        all_preds = []

        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            # batch["choice_list"] is [num_choices, batch_size]
            for i in range(len(batch["choice_list"][0])):
                all_answers.append(batch["choice_list"][batch["answer_id"][i]][i])

            choice_lists = batch.pop("choice_list")
            batch_len = len(batch["answer_id"])
            del batch["answer_id"]
            for key in batch:
                batch[key] = torch.stack(batch[key], dim=-1).to(device)

            with torch.no_grad():
                outputs = model(**batch)

                logits = outputs.logits

                # TODO: paralellize
                for i, logit in enumerate(logits):  # Assuming all are single tokens
                    choice_ids = torch.tensor(
                        [tokenizer.encode(" " + choice_lists[j][i], add_special_tokens=False)[0] for j in
                         range(len(choice_lists))])

                    MASK_INDEX = batch["input_ids"][i].tolist().index(MASK_ID)
                    probs = logit[MASK_INDEX].index_select(0, choice_ids.to(device))

                    max_ind = torch.argmax(probs)
                    all_preds.append(choice_lists[max_ind][i])

        return all_answers, all_preds
    else:
        raise NotImplementedError
