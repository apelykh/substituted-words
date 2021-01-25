import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from transformers.modeling_outputs import TokenClassifierOutput


class LSTMTokenClassifier(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 bidirectional: bool = True,
                 num_layers: int = 1,
                 batch_size: int = 512,
                 pretrained_embeddings: np.ndarray = None,
                 device: str = 'cuda'):
        """
        :param vocab_size: vocabulary size;
        :param embedding_dim: dimension of embedding vectors;
        :param hidden_dim: hidden size of the LSTM layer;
        :param bidirectional: if True, becomes a bidirectional LSTM;
        :param num_layers: number of recurrent layers;
        :param batch_size: batch size (needed for hidden state initialization);
        :param pretrained_embeddings: if passed, the embedding layer is initialized with pretrained weights;
        :param device: one of ('cpu', 'cuda'), the device that will host computations;
        """

        super(LSTMTokenClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        # needed only for hidden/cell state initialization
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device

        if pretrained_embeddings is not None:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                                freeze=False)
        else:
            self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=embedding_dim,
                                                padding_idx=0)
        self.hidden = None

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout()

        self.num_directions = 2 if bidirectional else 1
        # 2 output neurons for consistency with BertForTokenClassification
        self.linear = nn.Linear(self.num_directions * hidden_dim, 2)

    def _init_hidden(self):
        """
        Initialize LSTM hidden/cell states.
        """
        hidden_state = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim)

        hidden_state = Variable(hidden_state.cuda().to(self.device))
        cell_state = Variable(cell_state.cuda().to(self.device))

        return hidden_state, cell_state

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor) -> TokenClassifierOutput:
        """
        :param x: input data of dimensions [batch_size, seq_length];
        :param attention_mask: binary mask of dimensions [batch_size, seq_length]
            with 0 on padded token positions and 1 on active tokens;
        :return: outputs of a linear layer (raw, unnormalized scores for each class)
            wrapped in TokenClassifierOutput;
        """
        # reset LSTM hidden/cell states, should be done before every new batch.
        # Otherwise, a new batch will be treated as a continuation of a sequence
        self.hidden = self._init_hidden()

        _, padded_seq_len = x.size()

        out = self.word_embeddings(x)

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
