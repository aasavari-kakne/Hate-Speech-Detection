import transformers
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaConfig
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
from dataset_loader import COVID19TweetDataset, COVID19TweetDataset_unlabel
from classifiers import LSTMClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch.nn.functional as F

writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using Device: {}".format(device))

DATA_DIR = './data'
PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_LEN = 200
BATCH_SIZE = 16
RANDOM_SEED = 42
model_save_path = os.path.join(DATA_DIR, 'roberta_lstm_model_ea_weighted.ep1')
label_save_path = os.path.join(DATA_DIR, 'soft_labels.txt')

tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
unlabelled_dataset = os.path.join(DATA_DIR, 'processed_tweets_31.csv')

unlabelled_set = pd.read_csv(unlabelled_dataset, sep=',')
unlabelled_set = unlabelled_set[['processed_txt','id_str']]

def create_data_loader(data_set, tokenizer, max_len, batch_size):
    temp_data_set = COVID19TweetDataset_unlabel(data_set['processed_txt'].to_numpy(),
                                        data_set['id_str'].to_numpy(), max_len, tokenizer)

    return DataLoader(temp_data_set, batch_size=batch_size, drop_last=True)


# Creating data loaders
data_loader = {
    'unlabel': create_data_loader(unlabelled_set, tokenizer, MAX_LEN, BATCH_SIZE)
}

print('Data loaded into the data loader')

# Initializing model
print('Initializing the model')
roberta_config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
roberta_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=roberta_config)
model = LSTMClassifier(2, roberta_model, 512, 512, bidirectional=True, batch_size=BATCH_SIZE, device=device)
optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

# Loading model
print('Loading the model')
params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
model.load_state_dict(params['state_dict'])
model = model.to(device)

print('restore parameters of the optimizers')
optimizer.load_state_dict(params['optimizer'])

# Forward pass
def forward(model):

    #forward pass
    model.eval()
    val_predicted_labels = []
    val_tweet_ids = []
    val_tweet_text = []
    prob_neg_label = []
    prob_pos_label = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader['unlabel'])):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tweet_ids = batch['tweet_ids']
            tweet_text = batch['tweet_text']

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            outputs_softmax = F.softmax(outputs, dim=1)
            val_predicted_labels += list(predictions.detach().cpu().view(-1).numpy())
            prob_neg_label += list(outputs_softmax[:, 0].detach().cpu().view(-1).numpy())
            prob_pos_label += list(outputs_softmax[:, 1].detach().cpu().view(-1).numpy())
            val_tweet_ids += list(tweet_ids)
            val_tweet_text += list(tweet_text)


    #write the soft labels to a file
    print('Tweet text length: ', len(val_tweet_text))
    print('Tweet label length: ', len(val_predicted_labels))

    with open(label_save_path, 'w') as fout:
        fout.write('tweet_text'+'\t'+'predicted_label'+'\n')
        for i in range(0, len(val_predicted_labels)):
            fout.write(val_tweet_text[i]+'\t'+str(val_predicted_labels[i])+'\n')

forward(model)
