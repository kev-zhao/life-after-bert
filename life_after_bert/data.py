import json
import logging
import os
import sys
import random
import tqdm

import torch
from torch.utils.data import Dataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def load_olmpics_data(file_path, num_choices, num_samples=-1):
    """
    Adapted from https://github.com/alontalmor/oLMpics/blob/500bdfc5779736bd7258925f53516fb126fe3245/oLMpics
    /allennlp_models/dataset_readers/transformer_masked_lm_reader.py#L53
    """
    with open(file_path, "r") as data_file:
        item_jsons, questions, choice_lists, answer_ids = ([] for _ in range(4))  # Initialize to empty lists
        for line in data_file:
            item_jsons.append(json.loads(line.strip()))

    if num_samples != -1:
        item_jsons = random.sample(item_jsons, num_samples)

    for i, item_json in tqdm.tqdm(enumerate(item_jsons), total=len(item_jsons)):
        question_text = item_json["question"]["stem"]

        choice_label_to_id = {}
        choice_text_list = []
        choice_label_list = []

        any_correct = False

        for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
            choice_label = choice_item["label"]
            choice_label_to_id[choice_label] = choice_id
            choice_text = choice_item["text"]

            choice_text_list.append(choice_text)
            choice_label_list.append(choice_label)

            if item_json.get('answerKey') == choice_label:
                if any_correct:
                    raise ValueError("More than one correct answer found for {item_json}!")

                any_correct = True

        if not any_correct and 'answerKey' in item_json:
            raise ValueError("No correct answer found for {item_json}!")

        answer_id = choice_label_to_id.get(item_json.get("answerKey"))

        # If there are less than num_choices, pad choice_text_list with empty strings
        if len(choice_text_list) != num_choices:
            choice_text_list = (choice_text_list + num_choices * [''])[:num_choices]
            if answer_id is not None and answer_id >= num_choices:
                logging.warning(f"Skipping question with more than {num_choices} answers: {item_json}")
                continue

        questions.append(question_text)
        choice_lists.append(choice_text_list)
        answer_ids.append(answer_id)

    return questions, choice_lists, answer_ids


class MCDataset(Dataset):
    def __init__(self, questions, choices, answer_ids, tokenizer, mask_token=None, max_length=25):
        mask_token = mask_token if mask_token is not None else tokenizer.mask_token
        assert mask_token is not None, "mask_token must be provided if tokenizer.mask_token does not exist"
        questions = [question.replace('[MASK]', mask_token) for question in questions]
        out = tokenizer(questions, max_length=max_length, padding="max_length")
        self.input_ids = out["input_ids"]
        self.attention_mask = out["attention_mask"]
        self.questions = questions
        self.choices = choices
        self.answer_ids = answer_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "choice_list": self.choices[i],
            "answer_id": self.answer_ids[i],
        }


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = []

    for single_dict in batch:
        for key, value in single_dict.items():
            batch_dict[key].append(value)

    for key, value in batch_dict.items():
        if key != "choice_list":
            batch_dict[key] = torch.tensor(value)

    return batch_dict
