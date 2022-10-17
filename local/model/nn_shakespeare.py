"""RNN model in pytorch
References:
    [1] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas.
    Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
    [2] Reddi S, Charles Z, Zaheer M, et al.
    Adaptive Federated Optimization. ICML 2020.
    https://arxiv.org/pdf/2003.00295.pdf
"""
import torch.nn as nn
import torch

# fedprox中好像是100个隐藏单元

class Model(nn.Module):
    def __init__(self, output_features=80, embedding_dim=8, hidden_size=256):
        """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).

        Args:
            vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
                Defaults to 80.
            embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding,
                Defaults to 8.
            hidden_size (int, optional): the size of hidden layer. Defaults to 256.

        Returns:
            A `torch.nn.Module`.

        Examples:
            RNN_Shakespeare(
              (embeddings): Embedding(80, 8, padding_idx=0)
              (lstm): LSTM(8, 256, num_layers=2, batch_first=True)
              (fc): Linear(in_features=256, out_features=90, bias=True)
            ), total 819920 parameters
        """
        super().__init__()
        vocab_size = output_features
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output
