import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTMLayer(nn.Module):
    def __init__(
            self,
            input_size=200, hidden_size=200,
            num_layers=1, batch_size=256,
            bidirectional=False, inner_dropout=0.25,
            outer_droput=[0.25, 0.25]
    ):
        super(CustomLSTMLayer, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size,
            self.num_layers, batch_first=True,
            bidirectional=self.bidirectional
        )

    def forward(self, input):
        _, (ht, _) = self.lstm(input)
        return torch.squeeze(ht, 0)

    def init_hidden_size(self):
        cell_state = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            self.batch_size,
            self.hidden_size
        )

        hidden_state = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            self.batch_size,
            self.hidden_size
        )

        return (hidden_state, cell_state)


# this layer is used to convert the words into numbers
class CustomEmbeddingLayer(nn.Module):
    def __init__(
            self,
            vocab_size, embedding_size,
            pretrained_embeddings=None, freeze=False
    ):
        super(CustomEmbeddingLayer, self).__init__()

        if pretrained_embeddings is None:
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            rows, cols = pretrained_embeddings.shape
            self.embed = nn.Embedding(num_embeddings=rows, embedding_dim=cols, padding_idx=0)
            self.embed.weight.data.copy_(pretrained_embeddings)

        self.embed.weight.requires_grad = not freeze

    def forward(self, input):
        return self.embed(input)


class CustomFullyConnected(nn.Module):
    def __init__(self, hidden_size=200):
        super(CustomFullyConnected, self).__init__()

        self.fc1 = nn.Linear(hidden_size, 2)

    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        return output