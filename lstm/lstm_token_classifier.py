import torch
from torch.autograd import Variable
import torch.nn as nn
# -----------------------------------
from lstm.dataset import TextDataset
from torch.utils.data import DataLoader
from lstm.utils import create_glove_matrix
from transformers.modeling_outputs import TokenClassifierOutput
import sys
sys.path.insert(0, '..')
from model_trainer import GeneralModelTrainer


class LSTMTokenClassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 bidirectional=True,
                 num_layers=1,
                 batch_size=512,
                 pretrained_embeddings=None,
                 device='cuda'):

        super(LSTMTokenClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        if pretrained_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings).float(),
                                                                freeze=False)
        else:
            self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=embedding_dim,
                                                padding_idx=0)

        self.hidden = None
        self.dropout = nn.Dropout()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.num_directions = 2 if bidirectional else 1

        self.linear = nn.Linear(self.num_directions * hidden_dim, 2)

    def _init_hidden(self):
        hidden_state = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim)

        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        hidden_state = Variable(hidden_state)
        cell_state = Variable(cell_state)

        return hidden_state, cell_state

    def forward(self, x, attention_mask):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self._init_hidden()

        _, padded_seq_len = x.size()

        out = self.word_embeddings(x)
        # out = self.dropout(out)

        seq_lengths = torch.sum(attention_mask, dim=1, dtype=torch.int64)

        out = torch.nn.utils.rnn.pack_padded_sequence(out,
                                                      seq_lengths.cpu().int(),
                                                      batch_first=True,
                                                      enforce_sorted=False)
        out, self.hidden = self.lstm(out, self.hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out,
                                                        batch_first=True,
                                                        total_length=padded_seq_len)
        out = self.dropout(out)
        out = self.linear(out)

        # for consistency with BertForTokenClassification
        return TokenClassifierOutput(
            logits=out
        )


if __name__ == '__main__':
    data_dir = '../data'
    device = 'cuda'

    d = TextDataset(base_path=data_dir, split_name='train_small', max_len=120)

    d_loader = DataLoader(d,
                          batch_size=512,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None)

    word2id = d.word2id

    pretrained_embeddings = create_glove_matrix('../../glove.6B/glove.6B.100d.txt',
                                                word2id=word2id,
                                                embed_dim=100)

    model = LSTMTokenClassifier(vocab_size=len(word2id),
                                embedding_dim=100,
                                hidden_dim=200,
                                pretrained_embeddings=pretrained_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    loss_function = nn.CrossEntropyLoss()
    trainer = GeneralModelTrainer(model, loss_function, optimizer, device)
    trainer.fit(d_loader, d_loader, 0, 1)
