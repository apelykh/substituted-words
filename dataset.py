import os
import torch
import numpy as np
import collections
from torch.utils import data


class TextDataset(data.Dataset):
    def __init__(self, base_path: str, split_name: str, max_len: int = None,
                 vocab: list = None, freq_threshold: int = 10):
        self._read_gt_files(base_path, split_name)

        if not max_len:
            line_lengths = [len(line.split(' ')) for line in self.lines]
            max_len = int(np.max(line_lengths))
        self.max_len = max_len
        print('[DATASET] Maximum line length: {}'.format(max_len))

        self.vocab = vocab if vocab else self._create_vocabulary(freq_threshold=freq_threshold)
        print('[DATASET] Vocabulary size: {}'.format(len(self.vocab)))

        self.word2id = {word: i + 2 for i, word in enumerate(self.vocab)}
        self.unk_index = 1
        self.word2id['<UNK>'] = self.unk_index
        self.word2id['<PAD>'] = 0

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

    def _create_vocabulary(self, freq_threshold):
        words = [word for line in self.lines for word in self.preprocess_line(line)]
        word_counter = collections.Counter(words)
        vocab = [word for word, freq in word_counter.items() if freq > freq_threshold]

        return vocab

    def __getitem__(self, index: int):
        line = self.preprocess_line(self.lines[index])
        line_len = min(len(line), self.max_len)

        encoded_line = np.zeros(self.max_len)
        encoded_line[:line_len] = np.array([self.word2id.get(word, self.unk_index)
                                            for word in line[:line_len]])

        line_labels = np.zeros(self.max_len)
        line_labels[:line_len] = self.labels[index].split(' ')[:line_len]

        return torch.as_tensor(encoded_line, dtype=torch.long),\
            torch.as_tensor(line_labels, dtype=torch.long),\
            line_len

    def __len__(self) -> int:
        return len(self.lines)

    def get_vocab(self):
        return self.vocab


if __name__ == '__main__':
    d = train_dataset = TextDataset(base_path='./data',
                            split_name='train_small',
                            max_len=None)

    encoded_line, line_labels, line_len = d[0]
    print(encoded_line.size())
    print(line_labels.size())
    print(line_len)
