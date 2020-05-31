import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from gensim.models import KeyedVectors
from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
from torch.utils import data
from lstm_classifier import CustomLSTMLayer, CustomEmbeddingLayer, CustomFullyConnected
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from dataset_loader_lstm import SentenceDataLoader
import numpy as np
import os

#args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: {}".format(device))
DATA_DIR = 'data/'
DATA_DIR_GLOVE = 'data/glove.twitter.27B.50d_w2v.txt'
DATASET_PATH = 'data/tweets_dataset.tsv'
RANDOM_SEED = 42
BATCH_SIZE = 16


def index_sentences(sentence_list, global_word_dict, tokenizer):
    indexed_sentences =list(
        map(
            lambda x: torch.LongTensor([
                global_word_dict[token.text]
                if token.text in global_word_dict else 1
                for token in tokenizer(x.lower())
            ]),
            tqdm(sentence_list)
        )
    )
    return indexed_sentences

def pad_sentences(batch):
    max_batch_length = max(list(map(lambda x: x[0].size(0), batch)))
    padded_sentences = torch.LongTensor(
        list(
            map(
                lambda x: np.pad(x[0].numpy(), (0, max_batch_length-x[0].size(0)), 'constant', constant_values=0),
                batch
            )
        )
    )
    sentence_labels = torch.FloatTensor(list(map(lambda x: x[1], batch)))
    return (padded_sentences, sentence_labels)


def data_preprocessing():
    # data wrenching
    word_vectors = KeyedVectors.load_word2vec_format(DATA_DIR_GLOVE, binary=False)  # load the glove vectors
    tweets_data_orig = pd.read_csv(DATASET_PATH, sep='\t')
    tweets_data_orig.loc[tweets_data_orig['expert'] == 'none_of_the_above', 'expert'] = 0
    tweets_data_orig.loc[tweets_data_orig['expert'] != 0, 'expert'] = 1
    tweets_data_final = tweets_data_orig[['text.clean', 'expert', 'id']]

    # Splitting data into train/test sets
    train_set, test_set = train_test_split(tweets_data_final,
                                           test_size=0.1,
                                           random_state=RANDOM_SEED)

    val_set, test_set = train_test_split(test_set,
                                         test_size=0.5,
                                         random_state=RANDOM_SEED)

    # preprocessing
    tokenizer = Tokenizer(vocab=Vocab(strings=list(word_vectors.vocab.keys())))
    global_word_dict = {key.lower(): index for index, key in
                        enumerate(word_vectors.vocab.keys(), start=2)}  # Make a dictionary with [word]->index
    torch_X = index_sentences(train_set['text.clean'].values, global_word_dict, tokenizer)
    torch_Y = torch.tensor(train_set['expert'].to_numpy().astype('float64'))
    torch_X_dev = index_sentences(val_set['text.clean'].values, global_word_dict, tokenizer)
    torch_Y_dev = torch.tensor(val_set['expert'].to_numpy().astype('float64'))
    return torch_X, torch_Y, torch_X_dev, torch_Y_dev, global_word_dict, word_vectors

def model_(global_word_dict, word_vectors):

    # model vars
    LSTM_HIDDEN_UNITS = 25
    VOCAB_SIZE = len(global_word_dict)
    EMBEDDING_SIZE = 50

    # model
    model = nn.Sequential(
        CustomEmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            embedding_size=EMBEDDING_SIZE,
            pretrained_embeddings=torch.FloatTensor(word_vectors.vectors)  # find the correct code here
        ),
        CustomLSTMLayer(
            input_size=EMBEDDING_SIZE, hidden_size=LSTM_HIDDEN_UNITS,
            batch_size=BATCH_SIZE
        ),
        CustomFullyConnected(LSTM_HIDDEN_UNITS),
    )

    return model

def train(model, torch_X, torch_Y, torch_X_dev, torch_Y_dev):

    #hyper params
    LEARNING_RATE = 0.01
    EPOCHS = 10

    #training
    history = []
    train_dataset = SentenceDataLoader(torch_X, torch_Y)
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=pad_sentences,
        shuffle=True
    )

    val_dataset = SentenceDataLoader(torch_X_dev, torch_Y_dev)
    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=pad_sentences,
        shuffle=True
    )


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Set up the optimizer
    ce_loss = nn.BCEWithLogitsLoss().to(device) # Set up the loss


    for epoch in range(EPOCHS):
        # Set the progress bar up
        progress_bar = tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
        )

        # throw the model on the gpu
        model = model.to(device)

        avg_epoch_loss = []
        model.train()

        print('training starts')
        for index, batch in progress_bar:

            data_batch = batch[0]
            data_labels = torch.zeros(batch[0].size(0), 2)
            data_labels[range(batch[0].size(0)), batch[1].long()] = 1
            # Throw it on the gpu
            data_batch = data_batch.to(device)
            data_labels = data_labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # predict batch
            predicted = F.softmax(model(data_batch), dim=1)
            # Calculate the loss
            loss = ce_loss(predicted, data_labels)
            avg_epoch_loss.append(loss.item())
            loss.backward()

            # Update the weights
            optimizer.step()

            progress_bar.set_postfix(avg_loss=avg_epoch_loss[-1])

        model.eval()
        avg_epoch_loss_val = []
        predicted_proba = []
        dev_targets = []

        for val_batch in val_data_loader:
            val_data_batch = val_batch[0]
            val_data_batch = val_data_batch.to(device)
            val_data_labels = torch.zeros(val_batch[0].size(0), 2).to(device)
            val_data_labels[range(val_batch[0].size(0)), val_batch[1].long()] = 1

            predicted = F.softmax(model(val_data_batch), dim=1)
            loss_ = ce_loss(predicted, val_data_labels)
            avg_epoch_loss_val.append(loss_.item())

            predicted_proba.append(predicted[:, 1])
            dev_targets.append(val_batch[1])

        predicted_proba = torch.cat(predicted_proba, dim=0)
        dev_targets = torch.cat(dev_targets)
        predicted_labels = list(
            map(
                lambda x: 1 if x > 0.5 else 0,
                predicted_proba
                    .cpu()
                    .float()
                    .detach()
                    .numpy()
            )
        )

        print('Epoch: %d, Train Loss: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val F1:  %0.4f' % (epoch+1, np.mean(avg_epoch_loss),  np.mean(avg_epoch_loss_val),
                                                 accuracy_score(dev_targets.long().numpy(), predicted_labels),
                                                 f1_score(dev_targets.long().numpy(), predicted_labels)))

        # Save history
        history.append([np.mean(avg_epoch_loss), np.mean(avg_epoch_loss_val), accuracy_score(dev_targets.long().numpy(), predicted_labels), f1_score(dev_targets.long().numpy(), predicted_labels)])
        np.save(os.path.join(DATA_DIR, 'history_lstm.npy'), history)

def main():

    torch_X, torch_Y, torch_X_dev, torch_Y_dev, global_word_dict, word_vectors = data_preprocessing()
    model = model_(global_word_dict, word_vectors)
    train(model, torch_X, torch_Y, torch_X_dev, torch_Y_dev)

if __name__ == "__main__":
    main()