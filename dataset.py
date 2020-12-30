import os
import torch
import numpy as np
import collections
from torch.utils import data


class TextDataset(data.Dataset):
    def __init__(self, base_path: str, split_name: str, max_len: int = 100):
        src_text_path = os.path.join(base_path, '{}.src'.format(split_name))
        with open(src_text_path, 'r') as f:
            self.lines = f.readlines()

        labels_path = os.path.join(base_path, '{}.lbl'.format(split_name))
        with open(labels_path, 'r') as f:
            self.labels = f.readlines()

        self.max_len = max_len

        self.vocab = self._create_vocabulary(freq_threshold=12)

        self.word2id = {word: i + 2 for i, word in enumerate(self.vocab)}
        self.word2id['<UNK>'] = 1
        self.word2id['<PAD>'] = 0
        # self.word2id['<UNK>'] = len(self.word2id)
        # self.word2id['<PAD>'] = len(self.word2id)
        self.id2word = {i: word for word, i in self.word2id.items()}

    def _create_vocabulary(self, freq_threshold: int = 12):
        words = [word.lower() for line in self.lines for word in line.split(' ')]
        word_counter = collections.Counter(words)
        vocab = [word for word, freq in word_counter.items() if freq > freq_threshold]

        return vocab

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int):
        """
        Basic strategy: define a maximum line length, discard the rest of the line, pad if src line is shorter
        """
        full_line = self.lines[index].split(' ')
        line_len = min(len(full_line), self.max_len)

        encoded_line = np.zeros(self.max_len)
        unk_index = 1
        encoded_line[:line_len] = np.array([self.word2id.get(word, unk_index) for word in full_line[:line_len]])

        line_labels = np.zeros(self.max_len)
        line_labels[:line_len] = self.labels[index].split(' ')[:line_len]

        return torch.as_tensor(encoded_line, dtype=torch.long),\
               torch.as_tensor(line_labels, dtype=torch.long),\
               line_len

    def get_vocab(self):
        return self.vocab


if __name__ == '__main__':
    d = TextDataset('./data', 'val')

    print(d[954])
