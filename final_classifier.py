from torch import nn
import torch

class FinalClassifier(nn.Module):
    def __init__(self, num_classes, model):
        super(FinalClassifier, self).__init__()
        self.model = model
        self.cnn = nn.Conv2d(12, 4, kernel_size=(3, 768), padding=(1, 0))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax()
        

    def forward(self, input_ids, attention_mask):
        _, _, out_all_layers = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        # tuple of 13 tensors each BATCH_SIZE x 512 x 768

        num_layers = len(out_all_layers) - 1 # 13
        batch_size, max_len, hidden_dim = out_all_layers[0].size() # 4, 400, 768

        print(num_layers, batch_size, max_len, hidden_dim)

        out = torch.zeros([num_layers, batch_size, max_len, hidden_dim])

        for i in range(num_layers):
            out[i, :, :, :] = out_all_layers[i+1] 

        out.transpose(0, 1)

        print(out.size())


        # out = torch.cat(out[1:])       # shape becomes 12 x BATCH_SIZE X 400 x 768
        # out = out.transpose(0, 1)      # shape becomes BATCH_SIZE x 12 x 400 x 768
        # out = self.cnn(out)            # shape becomes BATCH_SIZE x 4 x 400 x 1
        # out = out.transpose(1, 3)      # shape becomes BATCH_SIZE x 1 x 400 x 4
        # out = self.maxpool()           # shape becomes BATCH_SIZE x 1 x 256 x 2
        # out.squeeze()                  # shape becomes BATCH_SIZE x 256 x 2
        # out = torch.flatten(out, start_dim=1, end_dim=2) # shape becomes BATCH_SIZE x 512
        # out = self.linear(out)         # shape becomes BATCH_SIZE x 2
        # out = self.softmax(out)        # shape becomes BATCH_SIZE x 2

        return out

