import os
import torch
import numpy as np
import unicodedata
from torch.utils import data
from transformers import BertTokenizer


class BertTextDataset(data.Dataset):
    def __init__(self, root_dir: str, split_name: str, max_len: int = 150):
        """
        :param root_dir: path to the data folder;
        :param split_name: one of ('train', 'dev', 'val'). split_name.src and split_name.lbl are expected
            to be found in root_dir and contain source text sequences and their per-token labels correspondingly;
        :param max_len: biggest acceptable sequence length, longer sequences will be truncated to max_len.
            If None, no sequences will be truncated;

        Note: maximum acceptable sequence length for Bert model is 512. As such, max_len should be <512 as words
        will be further split into subword units that will increase sequence length;
        """
        self._read_gt_files(root_dir, split_name)

        if not max_len:
            line_lengths = [len(line.split(' ')) for line in self.lines]
            max_len = int(np.max(line_lengths))
        self.max_len = max_len
        print('[DATASET] Maximum line length: {}'.format(max_len))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def _read_gt_files(self, base_path: str, split_name: str):
        src_text_path = os.path.join(base_path, '{}.src'.format(split_name))
        with open(src_text_path, 'r') as f:
            self.lines = f.readlines()

        labels_path = os.path.join(base_path, '{}.lbl'.format(split_name))
        with open(labels_path, 'r') as f:
            self.labels = f.readlines()

    @staticmethod
    def preprocess_line(line: str) -> list:
        """
        Tokenize the input line and remove non-ascii characters from each token. Tokens that contain only
        non-ascii characters will be replaced with [UNK] tag.

        :param line: input line;
        :return: sequence of processed tokens;
        """
        filtered_tokens = []

        for token in line.rstrip().split(' '):
            filtered_token = unicodedata.normalize('NFKD', token).\
                encode('ascii', errors='ignore').decode('ascii').lower()
            if filtered_token == '':
                filtered_token = '[UNK]'
            filtered_tokens.append(filtered_token)

        # TODO: remove ugliness
        assert len(filtered_tokens) == len(line.split(' '))

        return filtered_tokens

    def _adapt_labels_length(self, sequence: list, sequence_labels: list) -> list:
        """
        Extend word labels to subword parts.

        :param sequence: sequence of words before tokenization;
        :param sequence_labels: sequence of labels, one for each word;
        :return: adapted sequence of labels;
        """
        # label for [CLS] token
        labels = [0]

        for word, label in zip(sequence, sequence_labels):
            tokenized_word = self.tokenizer(word, add_special_tokens=False)
            n_subwords = len(tokenized_word['input_ids'])
            labels.extend([label] * n_subwords)

        # label for [SEP] token
        labels.append(0)

        return labels

    def __getitem__(self, index: int):
        """
        :return:
            1. truncated/padded sequence of token ids;
            2. corresponding sequence of token labels;
            3. binary mask with 0 on [PAD] token positions;
        """
        line = self.preprocess_line(self.lines[index])
        labels = self.preprocess_line(self.labels[index])

        encoded_input = self.tokenizer(line,
                                       is_split_into_words=True,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)

        line_len = min(len(labels), self.max_len)

        # some words are tokenized to subword units -> labels are extended to each subword part
        labels = self._adapt_labels_length(line, labels)
        adapted_labels = np.zeros(self.max_len)
        adapted_labels[:line_len] = labels[:line_len]

        # TODO: remove ugliness
        assert len(encoded_input["input_ids"]) == len(adapted_labels)

        return torch.as_tensor(encoded_input["input_ids"], dtype=torch.long),\
            torch.as_tensor(adapted_labels, dtype=torch.long),\
            torch.as_tensor(encoded_input['attention_mask'])

    def __len__(self) -> int:
        return len(self.lines)


if __name__ == '__main__':
    d = BertTextDataset(root_dir='./data',
                        split_name='dev',
                        max_len=20)

    for i in range(len(d)):
        # if i % 2000 == 0:
        #     print('{}/{}'.format(i, len(d)))

        encoded_line, line_labels, mask = d[i]

        print(encoded_line)
        print(line_labels)
        print(mask)
