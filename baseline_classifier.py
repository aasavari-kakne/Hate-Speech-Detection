from torch import nn


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
