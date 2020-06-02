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
    tweets_data_orig.loc[tweets_data_orig['expert'] == 'discussion_of_eastasian_prejudice', 'expert'] = 0
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
    torch_Y = torch.tensor(train_set['expert'].to_numpy().astype('int64'), dtype=torch.long)
    torch_X_dev = index_sentences(val_set['text.clean'].values, global_word_dict, tokenizer)
    torch_Y_dev = torch.tensor(val_set['expert'].to_numpy().astype('int64'), dtype=torch.long)
    return torch_X, torch_Y, torch_X_dev, torch_Y_dev, global_word_dict, word_vectors

def model_(global_word_dict, word_vectors):

    # model vars
    LSTM_HIDDEN_UNITS = 100
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
    LEARNING_RATE = 2e-5
    EPOCHS = 40

    #training
    best_val_f1 = float('-inf')
    corr_best_train_f1 = float('-inf') #best train F1 corresponding to best val F1 score
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
    class_weights = torch.FloatTensor([1.0, 2.663]).cuda()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights).to(device)

    for epoch in range(EPOCHS):
        # Set the progress bar up


        train_predicted_labels = []
        train_actual_labels = []

        progress_bar = tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
        )

        # put the model on the gpu
        model = model.to(device)

        avg_epoch_loss = []
        model.train()

        print('training starts')
        for index, batch in progress_bar:

            data_batch = batch[0]
            data_labels = batch[1]

            # Throw it on the gpu
            data_batch = data_batch.to(device)
            data_labels = data_labels.type(torch.long)
            data_labels = data_labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # predict batch
            predicted = F.softmax(model(data_batch), dim=1)
            _, predicted_labels = torch.max(predicted, dim=1)

            train_actual_labels += list(data_labels.detach().cpu().view(-1).numpy())
            train_predicted_labels += list(predicted_labels.detach().cpu().view(-1).numpy())

            # Calculate the loss
            loss = ce_loss(predicted, data_labels)
            avg_epoch_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            optimizer.step()
            progress_bar.set_postfix(avg_loss=avg_epoch_loss[-1])

        model.eval()
        avg_epoch_loss_val = []
        val_predicted_labels = []
        val_actual_labels = []
        predicted_proba = []
        dev_targets = []
        val_len = 0

        with torch.no_grad():
            for val_batch in val_data_loader:

                val_data_batch = val_batch[0].to(device)
                val_data_labels = val_batch[1].type(torch.long)
                val_data_labels = val_data_labels.to(device)

                predicted = F.softmax(model(val_data_batch), dim=1)
                _, predicted_labels = torch.max(predicted, dim=1)
                loss_ = ce_loss(predicted, val_data_labels)
                avg_epoch_loss_val.append(loss_.item())

                val_actual_labels += list(val_data_labels.detach().cpu().view(-1).numpy())
                val_predicted_labels += list(predicted_labels.detach().cpu().view(-1).numpy())

                predicted_proba.append(predicted[:, 1])
                dev_targets.append(val_batch[1])

                val_len+=len(val_batch[0])

            train_f1_score = f1_score(np.array(train_actual_labels), np.array(train_predicted_labels))
            val_f1_score = f1_score(np.array(val_actual_labels), np.array(val_predicted_labels))
            val_acc = np.sum(np.array(val_actual_labels) == np.array(val_predicted_labels)) / val_len

        if val_f1_score > best_val_f1:
            print('best val changed')
            best_val_f1 = val_f1_score
            corr_best_train_f1 = train_f1_score

        print('Epoch: %d, Train Loss: %0.4f, Train F1 score: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val F1:  %0.4f' % (epoch+1, np.mean(avg_epoch_loss), train_f1_score,  np.mean(avg_epoch_loss_val),
                                                                                               val_acc, val_f1_score))
        print('Best Val F1 score is: ', best_val_f1)

        # Save history
        history.append([np.mean(avg_epoch_loss), np.mean(avg_epoch_loss_val), val_f1_score, val_acc])
        np.save(os.path.join(DATA_DIR, 'history_lstm.npy'), history)

    print('Best Val F1 score: %0.4f Corresponding best Train F1 score: %0.4f' % (best_val_f1,  corr_best_train_f1))

def main():

    torch_X, torch_Y, torch_X_dev, torch_Y_dev, global_word_dict, word_vectors = data_preprocessing()
    model = model_(global_word_dict, word_vectors)
    train(model, torch_X, torch_Y, torch_X_dev, torch_Y_dev)


if __name__ == "__main__":
    main()