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
from dataset_loader import COVID19TweetDataset
from final_classifier import FinalClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random

# TODO: Add tensorboard logging 
writer = SummaryWriter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using Device: {}".format(device))
# Creating dataset
DATA_DIR = './data'
dataset_path_train = os.path.join(DATA_DIR, 'train.tsv')
dataset_path_test = os.path.join(DATA_DIR, 'test.tsv')
dataset_path_val = os.path.join(DATA_DIR, 'val.tsv')

tweets_train = pd.read_csv(dataset_path_train, sep='\t')
tweets_test = pd.read_csv(dataset_path_test, sep='\t')
tweets_val = pd.read_csv(dataset_path_val, sep='\t')

PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_LEN = 200
BATCH_SIZE = 32
NUM_EPOCHS = 20
RANDOM_SEED = 42

tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Changing dataframes into dictionary of COVID19TweetDataset objects
def create_data_loader(data_set, tokenizer, max_len, batch_size):
    temp_data_set = COVID19TweetDataset(data_set['text.clean'].to_numpy(), data_set['expert'].to_numpy(),
                                    data_set['id'].to_numpy(), max_len, tokenizer)

    return DataLoader(temp_data_set, batch_size=batch_size)

# Creating data loaders
data_loader = {
                'train': create_data_loader(tweets_train, tokenizer, MAX_LEN, BATCH_SIZE),
                'val': create_data_loader(tweets_val, tokenizer, MAX_LEN, BATCH_SIZE),
                'test': create_data_loader(tweets_test, tokenizer, MAX_LEN, BATCH_SIZE)
            }

# Initializing model
roberta_config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
roberta_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=roberta_config)

model = FinalClassifier(2, roberta_model)
model = model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(data_loader['train']) * NUM_EPOCHS)

class_weights = torch.FloatTensor([1.0, 2.088]).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

# Train loop
def train(num_epochs, model):
    history = []
    best_val_f1 = float('-inf')
    label_history = []
    tensorboard_time_train = 0
    tensorboard_time_val = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_arr = []
        train_predicted_labels = []
        train_actual_labels = []
        train_tweet_ids = []

        for step, batch in enumerate(tqdm(data_loader['train'])):
            tensorboard_time_train += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            tweet_ids = batch['tweet_ids']

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            train_actual_labels += list(labels.detach().cpu().view(-1).numpy())
            train_predicted_labels += list(predictions.detach().cpu().view(-1).numpy())
            train_tweet_ids += tweet_ids
            train_loss_arr.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f'Train Epoch = {epoch}, Step = {step}, Train Loss = {np.mean(train_loss_arr)}')
            
            writer.add_scalar('train_loss', np.mean(train_loss_arr), tensorboard_time_train)

        train_f1_score = f1_score(np.array(train_actual_labels), np.array(train_predicted_labels))
        train_acc = np.sum(np.array(train_actual_labels) == np.array(train_predicted_labels)) / len(tweets_train)
        writer.add_scalar('train_f1_score', train_f1_score, epoch)

        # Validation 
        model.eval()
        val_loss_arr = []
        val_predicted_labels = []
        val_actual_labels = []
        val_tweet_ids = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader['val'])):
                tensorboard_time_val += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                tweet_ids = batch['tweet_ids']

                outputs = model(input_ids, attention_mask)
                _, predictions = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)
                val_actual_labels += list(labels.detach().cpu().view(-1).numpy())
                val_predicted_labels += list(predictions.detach().cpu().view(-1).numpy())
                val_tweet_ids += tweet_ids

                val_loss_arr.append(loss.item())
                if step % 10 == 0:
                    print(f'Val Epoch = {epoch}, Step = {step}, Val Loss = {np.mean(val_loss_arr)}')
                
                writer.add_scalar('val_loss', np.mean(val_loss_arr), tensorboard_time_val)

            val_f1_score = f1_score(np.array(val_actual_labels), np.array(val_predicted_labels))
            val_acc = np.sum(np.array(val_actual_labels) == np.array(val_predicted_labels)) / len(tweets_val)
            writer.add_scalar('val_f1_score', val_f1_score, epoch)
        
        # If we get better validation f1, save the labels/tweet ids for error analysis
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            np.save(os.path.join(DATA_DIR, 'label_history.npy'), list(zip(val_actual_labels, 
                                                            val_predicted_labels, val_tweet_ids)))
            save(model, epoch, optimizer, np.mean(val_loss_arr), model_prefix='roberta_linear_baseline_model_weighted_loss')
        
        print(f'Epoch {epoch}')
        print('-' * 20)
        print(f'Train Loss = {np.mean(train_loss_arr)}, F-1 Score = {train_f1_score}, Acc = {train_acc}')
        print(f'Val Loss = {np.mean(val_loss_arr)}, F-1 Score = {val_f1_score}, Acc = {val_acc}')

        # Save history
        history.append([np.mean(train_loss_arr), np.mean(val_loss_arr), train_f1_score, val_f1_score])
        np.save(os.path.join(DATA_DIR, 'history.npy'), history)
        print("Best F-1 score on validation dataset is {}".format(best_val_f1))


def save(model, epoch, optimizer, loss, model_prefix='model_', root='/content/drive/My Drive/CS224u_Final_Project/.model'):
    path = Path(root) / (model_prefix + '.ep')
    if not path.parent.exists():
        path.parent.mkdir()

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss}, path)

# Call Train/Val Loop
print("Begin Training!")
train(NUM_EPOCHS, model)
