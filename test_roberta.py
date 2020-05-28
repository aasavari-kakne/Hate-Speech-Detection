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
from dataset_loader import COVID19Dataset
from baseline_classifier import BaselineClassifierLinear
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using Device: {}".format(device))
# Creating dataset
DATA_DIR = './data'
dataset_path = os.path.join(DATA_DIR, 'tweets_dataset.tsv')
tweets_data_orig = pd.read_csv(dataset_path, sep='\t')
tweets_data_orig.loc[tweets_data_orig['expert'] == 'none_of_the_above', 'expert'] = 0
tweets_data_orig.loc[tweets_data_orig['expert'] != 0, 'expert'] = 1
tweets_data_final = tweets_data_orig[['text.clean', 'expert']]
# print(tweets_data_final)

PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_LEN = 400
BATCH_SIZE = 64
RANDOM_SEED = 42

tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Splitting data into train/test sets
train_set, test_set = train_test_split(tweets_data_final,
                                       test_size=0.1,
                                       random_state=RANDOM_SEED)

val_set, test_set = train_test_split(test_set,
                                       test_size=0.5,
                                       random_state=RANDOM_SEED)


def create_data_loader(data_set, tokenizer, max_len, batch_size):
    temp_data_set = COVID19Dataset(data_set['text.clean'].to_numpy(), data_set['expert'].to_numpy(),
                                   max_len, tokenizer)

    return DataLoader(temp_data_set, batch_size=batch_size)

# Creating data loaders
data_loader = {
                'train': create_data_loader(train_set, tokenizer, MAX_LEN, BATCH_SIZE),
                'val': create_data_loader(val_set, tokenizer, MAX_LEN, BATCH_SIZE),
                'test': create_data_loader(test_set, tokenizer, MAX_LEN, BATCH_SIZE)
            }

# Initializing model
roberta_config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
roberta_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=roberta_config)

# last_hidden_state, pooler_output, hidden_states = roberta_model(input_ids=data['input_ids'],
#                                                                 attention_mask=data['attention_mask'])


model = BaselineClassifierLinear(2, roberta_model)
model = model.to(device)

NUM_EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(data_loader['train']) * NUM_EPOCHS)
criterion = nn.CrossEntropyLoss().to(device)


# Train loop
def train(num_epochs):
    for epoch in range(num_epochs):
        # Training
        train_loss_arr = []
        train_predicted_labels = []
        train_actual_labels = []

        for step, batch in enumerate(tqdm(data_loader['train'])):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            train_actual_labels += list(labels.detach().cpu().view(-1).numpy())
            train_predicted_labels += list(predictions.detach().cpu().view(-1).numpy())

            train_loss_arr.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f'Train Loss = {np.mean(train_loss_arr)}')

        train_f1_score = f1_score(np.array(train_actual_labels), np.array(train_predicted_labels))
        train_acc = np.sum(train_actual_labels == train_predicted_labels) / len(data_loader['train'])
        print(f'Epoch {epoch}')
        print('-' * 20)
        print(f'Train Loss = {np.mean(train_loss_arr)}, F-1 Score = {train_f1_score}, Acc = {train_acc}')


train(NUM_EPOCHS)
