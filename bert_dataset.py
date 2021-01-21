import os
import torch
import numpy as np
import collections
from torch.utils import data
from transformers import BertTokenizer


class BERTTextDataset(data.Dataset):
    def __init__(self, base_path: str, split_name: str, max_len: int = None):
        self._read_gt_files(base_path, split_name)

        if not max_len:
            line_lengths = [len(line.split(' ')) for line in self.lines]
            max_len = int(np.max(line_lengths))
        self.max_len = max_len
        print('[DATASET] Maximum line length: {}'.format(max_len))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

    def _read_gt_files(self, base_path, split_name):
        src_text_path = os.path.join(base_path, '{}.src'.format(split_name))
        with open(src_text_path, 'r') as f:
            self.lines = f.readlines()

        labels_path = os.path.join(base_path, '{}.lbl'.format(split_name))
        with open(labels_path, 'r') as f:
            self.labels = f.readlines()

    @staticmethod
    def preprocess_line(line: str) -> list:
        return line.rstrip().lower().split(' ')

    def _adapt_labels_length(self, sequence: list, sequence_labels: list):
        # TODO: properly account for special tokens in labels sequence;
        # tokenized_sentence = []
        labels = [0]

        for word, label in zip(sequence, sequence_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer(word, add_special_tokens=False)
            # tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list `n_subwords` times
            n_subwords = len(tokenized_word['input_ids'])
            labels.extend([label] * n_subwords)

        labels.append(0)

        return labels

    def __getitem__(self, index: int):
        line = self.preprocess_line(self.lines[index])
        labels = self.preprocess_line(self.labels[index])

        # some words are tokenized to subword units -> labels are extended to each subword part
        labels = self._adapt_labels_length(line, labels)

        encoded_input = self.tokenizer(line,
                                       is_split_into_words=True,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)

        line_len = min(len(labels), self.max_len)

        # encoded_line = np.zeros(self.max_len)
        # encoded_line[:line_len] = np.array([self.tokenizer.convert_tokens_to_ids(token)
        #                                     for token in tokenized_line[:line_len]])

        line_labels = np.zeros(self.max_len)
        line_labels[:line_len] = labels[:line_len]

        assert len(encoded_input["input_ids"]) == len(line_labels)

        return torch.as_tensor(encoded_input["input_ids"], dtype=torch.long),\
            torch.as_tensor(line_labels, dtype=torch.long),\
            torch.as_tensor(encoded_input['attention_mask'])

    def __len__(self) -> int:
        return len(self.lines)


if __name__ == '__main__':
    d = BERTTextDataset(base_path='./data',
                                        split_name='train_small',
                                        max_len=200)

    for i in range(len(d)):
        encoded_line, line_labels, mask = d[i]
        print(encoded_line)
        print(line_labels)
        print(mask)
