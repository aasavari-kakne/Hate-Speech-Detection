from torch import nn
import torch

class FinalClassifier(nn.Module):
    def __init__(self, num_classes, model):
        super(FinalClassifier, self).__init__()
        self.model = model
        self.cnn = nn.Conv2d(12, 4, kernel_size=(3, 768), padding=(1, 0))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Linear(200, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        _, _, out_all_layers = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        # tuple of 13 tensors each BATCH_SIZE x 200 x 768

        num_layers = len(out_all_layers) - 1 # 13 - 1 = 12 
        batch_size, max_len, hidden_dim = out_all_layers[0].size() # 32, 200, 768
        out = torch.zeros([num_layers, batch_size, max_len, hidden_dim], device=out_all_layers[0].device)
        for i in range(num_layers):
            out[i, :, :, :] = out_all_layers[i+1] 
        out = out.transpose(0, 1)                        # BATCH_SIZE x 12 x 200 x 768
        out = self.cnn(out)                              # BATCH_SIZE x 4 x 200 x 1
        out = out.squeeze()                              # BATCH_SIZE x 4 x 200
        out = self.maxpool(out)                          # BATCH_SIZE x 2 x 100
        out = torch.flatten(out, start_dim=1, end_dim=2) # BATCH_SIZE x 200
        out = self.linear(out)                           # BATCH_SIZE x 2
        out = self.dropout(out)                          # BATCH_SIZE X 2
        out = self.softmax(out)                          # BATCH_SIZE x 2

        return out

