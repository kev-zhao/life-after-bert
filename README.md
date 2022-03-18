# Life After BERT

## Introduction
In this work, we apply a variety of models to the oLMpics benchmark and psycholinguistic probing datasets to determine the role of different factors such as architecture, directionality,
size of the dataset, and pre-training objective on a model's linguistic capabilities.

## Installation

You can install the package via

```bash
pip install git+https://github.com/kev-zhao/life-after-bert
```

Or you can download the source code and install the package in editable mode

```bash
git clone https://github.com/kev-zhao/life-after-bert
cd life-after-bert
pip install -e .
```

## Run
Main.py only works for encoders for now (Thursday 3/10), see notebooks for other model architectures.
```bash
usage: main.py [-h] model_name task data_path num_choices

positional arguments:
  model_name   Identifier of any pretrained transformers.AutoModelForMaskedLM
  task         Type of task, among `oLMpics MLM`
  data_path    Path to jsonl file containing dataset questions
  num_choices  Number of answer choices for each question

optional arguments:
  -h, --help   show this help message and exit
```

### Example:
```bash
python main.py roberta-large "oLMpics MLM" \
    tests/data/oLMpics_age_comparison_dev.jsonl 2
```
*Expected output*: `Accuracy: 0.986` 

Read more [here](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3).
