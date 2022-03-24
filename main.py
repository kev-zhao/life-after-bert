import argparse

import numpy as np
import torch
import transformers

from life_after_bert.eval import evaluate_encoder, evaluate_decoder, evaluate_encoder_decoder
from life_after_bert.data import MCDataset


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="Identifier of any pretrained transformers.AutoModelForMaskedLM")
    parser.add_argument("--task", help="Type of task, among `oLMpics MLM`")  # TODO: add more tasks
    parser.add_argument("--data_path", help="Path to jsonl file containing dataset questions")  # TODO: Hug Hub dataset
    parser.add_argument("--num_choices", help="Number of answer choices for each question", type=int)

    return parser.parse_args()


def main(args):  # Copied from notebook for now
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    dataset = MCDataset.load_data("Age Comparison", 2, tokenizer)

    all_answers, all_preds = evaluate_encoder(model, tokenizer, args.task, dataset, device)
    print(f"Accuracy: {(np.array(all_answers) == np.array(all_preds)).mean()}")  # TODO: logger, print dataset name


if __name__ == '__main__':
    args = get_args()
    main(args)
