import torch
from torch.autograd import Variable
import torch.nn as nn
# -----------------------------------
from blstm.dataset import TextDataset
from torch.utils.data import DataLoader
from model_trainer import ModelTrainer
from utils import create_embedding_matrix


class LSTMTokenClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, bidirectional=True,
                 pretrained_embeddings=None):
        super(LSTMTokenClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = 512

        if pretrained_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings).float(),
                                                                freeze=False)
        else:
            self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=embedding_dim,
                                                padding_idx=0)

        self.hidden = None

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.num_directions = 2 if bidirectional else 1

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(self.num_directions * hidden_dim, 1)

    def _init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_state = torch.randn(self.num_directions * 1, self.batch_size, self.hidden_dim).cuda()
        cell_state = torch.randn(self.num_directions * 1, self.batch_size, self.hidden_dim).cuda()

        # if self.hparams.on_gpu:
        #     hidden_state = hidden_state.cuda()
        #     cell_state = cell_state.cuda()

        hidden_state = Variable(hidden_state)
        cell_state = Variable(cell_state)

        return hidden_state, cell_state

    def forward(self, x, seq_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self._init_hidden()

        batch_size, padded_seq_len = x.size()
        # print(seq_lengths[0], padded_seq_len)
        # print(x.size())

        out = self.word_embeddings(x)
        # print(out.size())
        # print(out.dtype)

        # start = time.time()
        out = torch.nn.utils.rnn.pack_padded_sequence(out,
                                                      seq_lengths,
                                                      batch_first=True,
                                                      enforce_sorted=False)
        # print(out.data.size())
        out, self.hidden = self.lstm(out, self.hidden)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out,
                                                        batch_first=True,
                                                        total_length=padded_seq_len)
        # print(out.size())
        # out = out.view(batch_size, padded_seq_len, 2, self.hidden_dim)
        # print(out.size())
        # sum states from forward/backward runs of the BLSTM
        # out = torch.sum(out, dim=2)
        # print('LSTM time: {}'.format(time.time() - start))
        # print(out.size())

        out = self.linear(out)
        # print(out.size())

        return out


if __name__ == '__main__':
    data_dir = './data'
    device = 'cuda'

    d = TextDataset(base_path=data_dir, split_name='train_small', max_len=None)

    d_loader = DataLoader(d,
                          batch_size=512,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None)

    vocab = d.get_vocab()

    word2id = d.word2id

    pretrained_embeddings = create_embedding_matrix('../glove.6B/glove.6B.100d.txt', word2id, len(word2id), 100)

    model = LSTMTokenClassifier(vocab_size=len(vocab) + 2,
                                embedding_dim=100,
                                hidden_dim=200,
                                pretrained_embeddings=pretrained_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # loss_function = nn.NLLLoss(ignore_index=0)
    loss_function = nn.BCEWithLogitsLoss()
    trainer = ModelTrainer(model, loss_function, optimizer, device)
    trainer.fit(d_loader, d_loader, 0, 1)

    # for batch in d_loader:
    #     tag_scores = model(batch[0].to(device))
    #     print(tag_scores)
    #     break

