import os
import torch
import numpy as np
import collections
import unicodedata
from torch.utils import data


class TextDataset(data.Dataset):
    def __init__(self, root_dir: str, split_name: str, max_line_len: int = 150,
                 vocab: list = None, freq_threshold: int = 10):
        """
        :param root_dir: path to the data folder;
        :param split_name: one of ('train', 'dev', 'val'). split_name.src and split_name.lbl are expected
            to be found in root_dir and contain source text sequences and their per-token labels correspondingly;
        :param max_line_len: biggest acceptable sequence length; longer sequences will be truncated to max_line_len.
            If None, no sequences will be truncated;
        :param vocab: if passed, the external vocabulary will be adopted. Otherwise, a new vocabulary
            will be created based on the current dataset;
        :param freq_threshold: minimum number of word occurrences in the dataset for
            the word to be included into the vocabulary;
        """
        self._read_gt_files(root_dir, split_name)

        if not max_line_len:
            line_lengths = [len(line.split(' ')) for line in self.lines]
            max_line_len = int(np.max(line_lengths))
        self.max_line_len = max_line_len
        print('[DATASET] Maximum line length: {}'.format(max_line_len))

        self.vocab = vocab if vocab else self._create_vocabulary(freq_threshold=freq_threshold)
        self.word2id = {word: i + 2 for i, word in enumerate(self.vocab)}
        self.unk_index = 1
        self.word2id['<UNK>'] = self.unk_index
        self.word2id['<PAD>'] = 0
        print('[DATASET] Vocabulary size: {}'.format(len(self.word2id)))

    def _read_gt_files(self, root_dir: str, split_name: str):
        src_text_path = os.path.join(root_dir, '{}.src'.format(split_name))
        with open(src_text_path, 'r') as f:
            self.lines = f.readlines()

        labels_path = os.path.join(root_dir, '{}.lbl'.format(split_name))
        with open(labels_path, 'r') as f:
            self.labels = f.readlines()

    def _create_vocabulary(self, freq_threshold: int) -> list:
        """
        Create a vocabulary of unique words from the dataset, filtering less frequent words out.

        :param freq_threshold: minimum number of word occurrences in the dataset for
            the word to be included into the vocabulary;
        :return: list of unique words from the dataset that passed the frequency threshold;
        """
        words = [word for line in self.lines for word in self.preprocess_line(line, mode='pre')]
        word_counter = collections.Counter(words)
        vocab = [word for word, freq in word_counter.items() if freq > freq_threshold]

        return vocab

    @staticmethod
    def preprocess_line(line: str, mode: str = 'post') -> list:
        """
        Tokenize the input line and remove non-ascii characters from each token.

        :param line: input line;
        :param mode: one of ('pre', 'post').
            When 'pre' is set, non-ascii characters are removed from the input line before tokenization.
            As such, tokens that consist only of non-ascii characters will be removed that may result in a
            different number of output tokens comparing line.split(). This mode should be used during a
            vocabulary creation stage to provide a cleaner vocabulary;

            When 'post' is set, the input line will be first split and then each token will be normalized
            separately. Tokens that contain only non-ascii characters will be replaced with [UNK] tag,
            preserving the number of output tokens. This mode should be used for inference;

        :return: sequence of processed tokens;
        """
        filtered_tokens = []

        if mode == 'pre':
            line = unicodedata.normalize('NFKD', line). \
                encode('ascii', errors='ignore').decode('ascii')
            return line.rstrip().lower().split(' ')
        elif mode == 'post':
            for token in line.rstrip().split(' '):
                filtered_token = unicodedata.normalize('NFKD', token). \
                    encode('ascii', errors='ignore').decode('ascii').lower()
                if filtered_token == '':
                    filtered_token = '<UNK>'
                filtered_tokens.append(filtered_token)
            return filtered_tokens
        else:
            raise ValueError("Invalid line pre-processing mode")

    def __getitem__(self, index: int):
        """
        :param index: index of a sample (one sample is a line);
        :return:
            1. truncated/padded sequence of token ids;
            2. corresponding sequence of token labels;
            3. binary mask with 0 on [PAD] token positions;
        """
        line = self.preprocess_line(self.lines[index])
        line_len = min(len(line), self.max_line_len)

        encoded_line = np.zeros(self.max_line_len)
        encoded_line[:line_len] = np.array([self.word2id.get(word, self.unk_index)
                                            for word in line[:line_len]])

        line_labels = np.zeros(self.max_line_len)
        line_labels[:line_len] = self.labels[index].split(' ')[:line_len]

        mask = np.zeros(self.max_line_len)
        mask[:line_len] = 1

        return torch.as_tensor(encoded_line, dtype=torch.long),\
            torch.as_tensor(line_labels, dtype=torch.long),\
            torch.as_tensor(mask, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.lines)

    def get_vocab(self):
        return self.vocab
