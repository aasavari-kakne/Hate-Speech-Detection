import torch
from torch.utils.data import Dataset, DataLoader


class COVID19TweetDataset(Dataset):
    def __init__(self, tweet_text, labels, tweet_ids, max_len, tokenizer, batch_size=32):
        self.tweet_text = tweet_text
        self.labels = labels
        self.tweet_ids = tweet_ids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.tweet_text)

    def __getitem__(self, item):
        tweet = str(self.tweet_text[item])
        label = self.labels[item]
        tweet_id = str(self.tweet_ids[item])

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'tweet_ids': tweet_id
        }


class COVID19TweetDataset_unlabel(Dataset):

    def __init__(self, tweet_text, tweet_ids, max_len, tokenizer, batch_size=32):
        self.tweet_text = tweet_text
        self.tweet_ids = tweet_ids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.tweet_text)

    def __getitem__(self, item):
        tweet = str(self.tweet_text[item])
        tweet_id = str(self.tweet_ids[item])

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tweet_ids': tweet_id
        }
