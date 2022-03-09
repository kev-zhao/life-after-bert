import unittest

import transformers

from life_after_bert.data import load_olmpics_data, MCDataset


class TestOLMpicsData(unittest.TestCase):
    def test_data(self):
        questions, choice_lists, answer_ids = load_olmpics_data("tests/data/oLMpics_age_comparison_train.jsonl", 2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = MCDataset(questions, choice_lists, answer_ids, tokenizer)
        print(dataset)
        print(dataset[0])
        self.assertTrue(True)


if __name__ == "__main__":
    test = TestOLMpicsData()
    test.test_data()
