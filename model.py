import torch
import torch.nn as nn
import torch.nn.functional as F
# -----------------------------------
from dataset import TextDataset
from torch.utils.data import DataLoader
from model_trainer import ModelTrainer


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
                            num_layers=3,
                            batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.word_embeddings(x)
        # print(out.size())
        out, _ = self.lstm(out)
        # print(out.size())

        out = self.linear(out)
        # print(out.size())
        # out = F.log_softmax(out, dim=1)
        # out = F.sigmoid(out)

        return out


if __name__ == '__main__':
    data_dir = './data'
    device = 'cuda'

    d = TextDataset(base_path=data_dir, split_name='train', max_len=100)

    d_loader = DataLoader(d,
                          batch_size=512,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None)

    vocab = d.get_vocab()
    print(len(vocab))

    model = WordSubstitutionDetector(vocab_size=len(vocab) + 2, embedding_dim=100, hidden_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # loss_function = nn.NLLLoss(ignore_index=2228)
    loss_function = nn.BCEWithLogitsLoss()
    trainer = ModelTrainer(model, loss_function, optimizer, device)
    trainer.fit(d_loader, d_loader, 0, 30)

    # for batch in d_loader:
    #     tag_scores = model(batch[0].to(device))
    #     print(tag_scores)
    #     break

