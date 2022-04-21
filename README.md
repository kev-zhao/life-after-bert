# Life After BERT

## Introduction
In this work, we apply a variety of models to the oLMpics benchmark and psycholinguistic probing datasets to determine the role of different factors such as architecture, directionality,
size of the dataset, and pre-training objective on a model's linguistic capabilities.

## Installation

You can install the package via

```bash
pip install git+https://github.com/kev-zhao/life-after-bert
```

Or **(recommended)** you can download the source code and install the package in editable mode:

```bash
git clone https://github.com/kev-zhao/life-after-bert
cd life-after-bert
pip install -e .
```

## Run
```bash
usage: main.py [-h] [--model_name MODEL_NAME] [--model_arch MODEL_ARCH] [--mask_token MASK_TOKEN] [--task TASK]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Identifier of any pretrained HuggingFace model
  --model_arch MODEL_ARCH
                        Model architecture, among `encoder`, `decoder`, and `encoder decoder`
  --mask_token MASK_TOKEN
                        Tokenizer mask token (string), if different from default. Mainly used for GPT2 ("[MASK]") and T5 ("<extra_id_0>").
  --task TASK           Type of task, among `oLMpics MLM`

```

### Example:
```bash
python main.py \
    --model_name roberta-large \
    --model_arch encoder
```
**Expected output**: 
```
Accuracy on Age Comparison: 0.986
Accuracy on Always Never: 0.1357142857142857
Accuracy on Antonym Negation: 0.746
Accuracy on Multihop Composition: 0.28
Accuracy on Size Comparison: 0.874
Accuracy on Taxonomy Conjunction: 0.4540901502504174
``` 

### Tested Models:
* BERT
* RoBERTa
* DistilBERT
* T5
* BART (due to the HuggingFace implementation, this works as an encoder instead of encoder-decoder)
* Pegasus (Seq2Seq model, not included in paper)
* GPT2
