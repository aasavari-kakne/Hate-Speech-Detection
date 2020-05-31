import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class BaselineClassifierLinear(nn.Module):
    def __init__(self, num_classes, model):
        super(BaselineClassifierLinear, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooler_output, _ = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)
        out = self.dropout(pooler_output)
        out = self.linear(out)

        return out

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, model, lstm_hidden_size, linear_hidden_size, bidirectional,
                    batch_size, device):
        super(LSTMClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.2)
        self.lstm_hidden_size = lstm_hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size
        self.device = device
        self.relu = nn.ReLU()
        self.hidden = self.init_hidden(batch_size=batch_size)
        self.lstm = nn.LSTM(self.model.config.hidden_size,
                            lstm_hidden_size, 
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=0.3)
        self.linear1 = nn.Linear(lstm_hidden_size * self.num_directions, linear_hidden_size)
        self.linear2 = nn.Linear(linear_hidden_size, num_classes)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_directions, batch_size, self.lstm_hidden_size)).to(self.device),
                Variable(torch.zeros(self.num_directions, batch_size, self.lstm_hidden_size)).to(self.device))

        return h, c

    def forward(self, input_ids, attention_mask):
        h_0, c_0 = self.init_hidden(self.batch_size)
        last_hidden_output, pooler_output, _ = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)
       
        sent_lens = torch.sum(input_ids != 1, dim=1)
        # print(last_hidden_output.shape)
        last_hidden_output_padded = pack_padded_sequence(last_hidden_output, sent_lens, batch_first=True, enforce_sorted=False)
        # print(last_hidden_output_padded.data.shape)
        lstm_output, (h_n, c_n) = self.lstm(last_hidden_output_padded, (h_0, c_0))
        # output_unpacked, output_lengths = pad_packed_sequence(lstm_output, batch_first=True)
        cat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # print(h_n.shape)
        # print(output_unpacked.shape)
        # out = output_unpacked[:, -1, :]
        # rel = self.relu(cat)
        out = self.linear1(cat)
        out = self.dropout(out)
        out = self.linear2(out)

        return out