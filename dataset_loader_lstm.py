from torch.utils import data

class SentenceDataLoader(data.Dataset):
    def __init__(self, train_data, train_labels):
        super(SentenceDataLoader, self).__init__()

        self.X = train_data
        self.Y = train_labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])
