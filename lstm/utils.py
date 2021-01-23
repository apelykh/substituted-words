import numpy as np


def create_glove_matrix(glove_path: str, word2id: dict, embed_dim: int = 100):
    """
    Parse GloVe txt file and fetch embeddings for words from word2id.

    :param glove_path: path to GloVe txt file;
    :param word2id: word-index mapping;
    :param embed_dim: length of embedding vectors;
    :return: matrix of size [len(word2id), embed_dim] with GloVe embeddings that correspond
        to the words from word2id;
    """
    embed_mat = np.random.rand(len(word2id), embed_dim)
    words_embedded = 0

    with open(glove_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]

            if word in word2id:
                word_id = word2id[word]
                embed_mat[word_id] = np.asarray(line_split[1:], dtype='float32')
                words_embedded += 1

    print('GloVe embeddings found for {}/{} tokens'.format(words_embedded, len(word2id)))

    return embed_mat
