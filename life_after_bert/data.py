import json
import logging
import os
import sys
import random
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


class MCDataset(Dataset):
    """
    Dataset for multiple choice (MC) tasks (tested on oLMpics and Ettinger but should work for all MC tasks)
    Can be initialized either with `MCDataset().load_data()` from a file with the oLMpics jsonl format,
    Or by directly with the constructor
    """
    TASK_TO_FILENAME = {
        "Age Comparison": "oLMpics_age_comparison_dev.jsonl",
        "Always Never": "oLMpics_always_never_dev.jsonl",
        "Antonym Negation": "oLMpics_antonym_negation_dev.jsonl",
        "Multihop Composition": "oLMpics_multihop_composition_dev.jsonl",
        "Size Comparison": "oLMpics_size_comparison_dev.jsonl",
        "Taxonomy Conjunction": "oLMpics_taxonomy_conjunction_dev.jsonl"
    }

    def __init__(self, questions, choices, answer_ids, num_choices, tokenizer, max_length=26):
        """
        Args:
            questions: list of questions, where each question is a string with a "[MASK]" token
            choices: list of list of choices (string); i.e. shape of 500 x 3, if there are 500 questions with 3 choices each
            answer_ids: index of the correct choice (integer) for each question
            num_choices: the number of choices (integer)  # TODO: replace with len(choices[0])
            tokenizer: HuggingFace tokenizer (to tokenize `questions` and `choices`)
            max_length: the maximum length of all questions, questions with more tokens than max_length are are truncated
        """
        assert tokenizer.mask_token is not None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Setting tokenizer's pad_token to eos_token")

        questions = [question.replace('[MASK]', tokenizer.mask_token) for question in questions]
        out = tokenizer(questions, max_length=max_length, padding="max_length", return_tensors="pt")
        self.input_ids = out["input_ids"]
        self.attention_mask = out["attention_mask"]
        self.answer_ids = torch.tensor(answer_ids)
        self.num_choices = num_choices

        tokenized_choices = []
        truncated_tokens = 0

        for i, curr_choices in enumerate(choices):
            output = tokenizer([" " + curr_choice for curr_choice in curr_choices], add_special_tokens=False, return_tensors="pt", max_length=1,
                               truncation=True, return_overflowing_tokens=True)

            # TODO: explain below lines with comments
            mask = torch.cat([torch.ones(1, dtype=int),
                              output.overflow_to_sample_mapping[1:] - output.overflow_to_sample_mapping[:-1]])
            choice_input_ids = torch.index_select(output.input_ids, 0, torch.nonzero(mask).flatten())
            truncated_tokens += len(mask) - len(choice_input_ids)
            tokenized_choices.append(choice_input_ids.squeeze(1))

        if truncated_tokens > 0:
            logger.warning(f"Truncated {truncated_tokens} tokens from answer choices.")

        self.choice_ids = tokenized_choices

    @classmethod
    def load_data(cls, task_name_or_path, num_choices, tokenizer, num_samples=-1):
        """
        Creates MCDataset from .jsonl file

        Args:
            task_name_or_path: Either an oLMpics task name (see all options from MCDataset.TASK_TO_FILENAME)
                               Or the path to a .jsonl file
            num_choices: Number of answer choices per question
            tokenizer: HuggingFace tokenizer
            num_samples: Number of samples to take for debugging, default value of -1 will load all samples

        Returns: MCDataset
        """
        if task_name_or_path in cls.TASK_TO_FILENAME.keys():
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tests", "data")
            task_name_or_path = os.path.join(data_dir, cls.TASK_TO_FILENAME[task_name_or_path])
            logger.info(f"Loading jsonl file from {task_name_or_path}")

        assert task_name_or_path.endswith(".jsonl"), "This function only works for files " \
                                                                           "with oLMpics jsonl format. Please " \
                                                                           "change the file extension to " \
                                                                           "`.jsonl` to resolve this error."

        """ The below code is adapted from 
        https://github.com/alontalmor/oLMpics/blob/500bdfc5779736bd7258925f53516fb126fe3245/oLMpics
        /allennlp_models/dataset_readers/transformer_masked_lm_reader.py#L53 """
        with open(task_name_or_path, "r") as data_file:
            item_jsons, questions, choice_lists, answer_ids = ([] for _ in range(4))  # Initialize to empty lists
            for line in data_file:
                item_jsons.append(json.loads(line.strip()))

        if num_samples != -1:
            item_jsons = random.sample(item_jsons, num_samples)

        for i, item_json in tqdm(enumerate(item_jsons), total=len(item_jsons), disable=True):
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

            if not any_correct:
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

        return cls(questions, choice_lists, answer_ids, num_choices, tokenizer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "choice_ids": self.choice_ids[i],
            "answer_id": self.answer_ids[i],
        }


def collate_fn(batch):
    """ Collate function for using MCDataset with torch.utils.data.DataLoader """
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = []

    for single_dict in batch:
        for key, value in single_dict.items():
            batch_dict[key].append(value)

    for key, value in batch_dict.items():
        batch_dict[key] = torch.stack(value, dim=0)

    return batch_dict
