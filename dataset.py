import os
import torch
import numpy as np
import collections
from torch.utils import data


class TextDataset(data.Dataset):
    def __init__(self, base_path: str, split_name: str, max_len: int = None,
                 vocab: list = None, freq_threshold: int = 10):

        src_text_path = os.path.join(base_path, '{}.tgt'.format(split_name))
        with open(src_text_path, 'r') as f:
            self.lines = f.readlines()

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

        # leaving 1 <PAD> token at the end as a target for the last word
        encoded_line = np.zeros(self.max_len + 1)
        encoded_line[:line_len] = np.array([self.word2id.get(word, self.unk_index)
                                            for word in line[:line_len]])

        targets = np.zeros(self.max_len + 1)
        targets[:line_len] = encoded_line[1:line_len + 1]

        return torch.as_tensor(encoded_line, dtype=torch.long),\
            torch.as_tensor(targets, dtype=torch.long),\
            line_len

    def __len__(self) -> int:
        return len(self.lines)

    def get_vocab(self):
        return self.vocab


if __name__ == '__main__':
    d = train_dataset = TextDataset(base_path='./data',
                                    split_name='dev',
                                    max_len=None,
                                    vocab=None,
                                    freq_threshold=10)

    encoded_line, targets, line_len = d[0]
    print(encoded_line.size())
    print(encoded_line)
    print(targets.size())
    print(targets)
    print(line_len)
