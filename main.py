import argparse
import logging
import os, sys

import torch
import transformers

from life_after_bert import LaBEvaluator


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))

ARCH_TO_CLASS = {
    "encoder": transformers.AutoModelForMaskedLM,
    "decoder": transformers.AutoModelForCausalLM,
    "encoder-decoder": transformers.T5ForConditionalGeneration
}

TASK_INFOS = [
    ("Age Comparison", 2),
    ("Always Never", 5),
    ("Antonym Negation", 2),
    ("Multihop Composition", 3),
    ("Size Comparison", 2),
    ("Taxonomy Conjunction", 3)
]


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True, help="Identifier of any pretrained HuggingFace model")
    parser.add_argument("--model_arch", required=True, help="Model architecture, among `encoder`, `decoder`, and `encoder-decoder`")
    parser.add_argument(
        "--mask_token",
        help='Tokenizer mask token (string), if different from default. '
             'Mainly used for GPT2 ("[MASK]") and T5 ("<extra_id_0>").'
    )

    return parser.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ARCH_TO_CLASS[args.model_arch].from_pretrained(args.model_name)
    if args.mask_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
        assert tokenizer.mask_token is not None
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, mask_token=args.mask_token)

    evaluator = LaBEvaluator()
    task_accs = evaluator.evaluate(model, tokenizer, TASK_INFOS, model_arch=args.model_arch, device=device)
    logger.info(task_accs)


if __name__ == '__main__':
    args = get_args()
    main(args)
