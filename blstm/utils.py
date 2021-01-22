import numpy as np
from blstm.dataset import TextDataset


def create_embedding_matrix(glove_path, word2id, vocab_size, embed_dim):
    # Initialize embeddings matrix to handle unknown words
    embed_mat = np.random.rand(vocab_size, embed_dim)
    words_embedded = 0

    with open(glove_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]

            if word in word2id:
                word_id = word2id[word]
                embed_mat[word_id] = np.asarray(line_split[1:], dtype='float32')
                words_embedded += 1

    print('Embeddings found for {}/{} tokens'.format(words_embedded, len(word2id)))

    return embed_mat


if __name__ == '__main__':
    train_dataset = TextDataset(base_path='../data',
                                split_name='train_small',
                                max_len=None)
    word2id = train_dataset.word2id
    pretrained_embeddings = create_embedding_matrix('../../glove.6B/glove.6B.100d.txt', word2id, len(word2id), 100)
