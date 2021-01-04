import torch
import torch.nn as nn
import torch.nn.functional as F
# -----------------------------------
from dataset import TextDataset
from torch.utils.data import DataLoader
from model_trainer import ModelTrainer
import time


class WordSubstitutionDetector(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordSubstitutionDetector, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_lengths):
        batch_size, seq_len = x.size()
        # print(x.size())

        out = self.word_embeddings(x)
        # print(out.size())

        # start = time.time()
        out = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(out)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=seq_len)
        # print('LSTM time: {}'.format(time.time() - start))
        # print(out.size())

        out = self.linear(out)
        # print(out.size())
        # out = F.log_softmax(out, dim=1)
        # out = F.sigmoid(out)

        return out


if __name__ == '__main__':
    data_dir = './data'
    device = 'cuda'

    d = TextDataset(base_path=data_dir, split_name='train_small', max_len=None)

    d_loader = DataLoader(d,
                          batch_size=1563,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None)

    vocab = d.get_vocab()

    model = WordSubstitutionDetector(vocab_size=len(vocab) + 2, embedding_dim=100, hidden_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # loss_function = nn.NLLLoss(ignore_index=2228)
    loss_function = nn.BCEWithLogitsLoss()
    trainer = ModelTrainer(model, loss_function, optimizer, device)
    trainer.fit(d_loader, d_loader, 0, 1)

    # for batch in d_loader:
    #     tag_scores = model(batch[0].to(device))
    #     print(tag_scores)
    #     break

