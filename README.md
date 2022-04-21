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
usage: main.py [-h] --model_name MODEL_NAME --model_arch MODEL_ARCH [--mask_token MASK_TOKEN]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Identifier of any pretrained HuggingFace model
  --model_arch MODEL_ARCH
                        Model architecture, among `encoder`, `decoder`, and `encoder-decoder`
  --mask_token MASK_TOKEN
                        Tokenizer mask token (string), if different from default. Mainly used for GPT2 ("[MASK]") and T5 ("<extra_id_0>").
```

### Example:
```bash
python main.py \
    --model_name roberta-large \
    --model_arch encoder
```
**Expected output**: 
```
{
    'Age Comparison': 0.986, 
    'Always Never': 0.1357142857142857, 
    'Antonym Negation': 0.744, 
    'Multihop Composition': 0.28, 
    'Size Comparison': 0.874, 
    'Taxonomy Conjunction': 0.4540901502504174
}
``` 

### Python Examples:
```python
from datasets import load_dataset
import torch, transformers
import life_after_bert

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use an encoder model
model = transformers.AutoModelForMaskedLM.from_pretrained("roberta-large")
tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-large")

# Evaluate on a HuggingFace dataset
hub_dataset = load_dataset("KevinZ/oLMpics", "Age_Comparison")["test"]
dataset = life_after_bert.MCDataset(hub_dataset["stem"], hub_dataset["choices"], hub_dataset["answerKey"], num_choices=2, tokenizer=tokenizer)

accuracy, (all_answers, all_preds) = life_after_bert.evaluate_encoder(model, tokenizer, dataset, device=device)
print(f"{model.config._name_or_path} accuracy: {accuracy} on Age Comparison task")
```
```python
# Use a decoder model
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large", mask_token="[MASK]")  # Causal LM's don't have mask tokens by default

# Evaluate on a custom dataset from a jsonl file
dataset = life_after_bert.MCDataset.load_data("../tests/data/oLMpics_always_never_dev.jsonl", num_choices=5, tokenizer=tokenizer)

accuracy, (all_answers, all_preds) = life_after_bert.evaluate_decoder(model, tokenizer, dataset, device=device)
print(f"{model.config._name_or_path} accuracy: {accuracy} on Always Never task")
```
```python
# Or evaluate on multiple tasks
evaluator = life_after_bert.LaBEvaluator()
task_infos = [
    ("Antonym Negation", 2),  # You can use the name of an oLMpics task
    ("Taxonomy Conjunction", 3),
    ("../tests/data/oLMpics_multihop_composition_dev.jsonl", 3),  # Or pass in the file paths
    ("../tests/data/oLMpics_size_comparison_dev.jsonl", 2),
]

# Use an encoder-decoder model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-large")
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-large", mask_token="<extra_id_0>")

task_accs = evaluator.evaluate(model, tokenizer, task_infos, model_arch="encoder-decoder", device=device)
print(task_accs)
```

### Tested Models:
* BERT
* RoBERTa
* DistilBERT
* T5
* BART (due to the HuggingFace implementation, this works as an encoder instead of encoder-decoder)
* Pegasus (Seq2Seq model, not included in paper)
* GPT2
