# -*- coding: utf-8 -*-
"""

A RNN classifier applied to AG_NEWS dataset

Download dataset:
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

"""

import torch

from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch import nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPER-PARAMETERS
MAX_WORDS = 25
EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EMBEDDING_DIM = 100
HIDDEN_DIM = 64

######################################################################
# Read dataset files 
# ------------------


train_data = pd.read_csv('ag-news-classification-dataset/train.csv')
test_data = pd.read_csv('ag-news-classification-dataset/test.csv')

######################################################################
# Data processing 
# -----------------------------


tokenizer = get_tokenizer("basic_english")

# All texts are truncated and padded to MAX_WORDS tokens
def collate_batch(batch):
    Y, X = list(zip(*batch))
    Y = torch.tensor(Y) - 1 # Target names in range [0,1,2,3] instead of [1,2,3,4]
    X = [vocab(tokenizer(text)) for text in X]
    # Bringing all samples to MAX_WORDS length. Shorter texts are padded with <PAD> sequences, longer texts are truncated.
    X = [tokens+([vocab['<PAD>']]* (MAX_WORDS-len(tokens))) if len(tokens)<MAX_WORDS else tokens[:MAX_WORDS] for tokens in X]
    return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device) 

train_dataset = [(label,train_data['Title'][i] + ' ' + train_data['Description'][i]) for i,label in enumerate(train_data['Class Index'])]
test_dataset = [(label,test_data['Title'][i] + ' ' + test_data['Description'][i]) for i,label in enumerate(test_data['Class Index'])]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_batch)

target_classes = ["World", "Sports", "Business", "Sci/Tech"]

def build_vocabulary(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

# Vocabulary includes all tokens with at least 10 occurrences in the texts
# Special tokens <PAD> and <UNK> are used for padding sequences and unknown words respectively
vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=10, specials=["<PAD>","<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

######################################################################
# Define the model
# ----------------


class model(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):
        super(model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings)
        logits = self.linear(output[:,-1])  # The last output of RNN is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs
    
######################################################################
# Initiate an instance of the model
# ---------------------------------


classifier = model(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, len(target_classes)).to(device)
# Define loss function and opimization algorithm
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([param for param in classifier.parameters() if param.requires_grad == True],lr=LEARNING_RATE)

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('\nModel:')
print(classifier)
print('Total parameters: ',count_parameters(classifier))
print('\n\n')

######################################################################
# Define functions to train and evaluate the model
# ------------------------------------------------


def EvaluateModel(model, loss_fn, val_loader):
    model.eval()
    with torch.no_grad():
        Y_actual, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_actual.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_actual = torch.cat(Y_actual)
        Y_preds = torch.cat(Y_preds)
    
    # Returns mean loss, actual labels, predicted labels 
    return torch.tensor(losses).mean(), Y_actual.detach().cpu().numpy(), Y_preds.detach().cpu().numpy()


def TrainModel(model, loss_fn, optimizer, train_loader, epochs):
    for i in range(1, epochs+1):
        model.train()
        print('Epoch:',i)
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        

TrainModel(classifier, loss_fn, optimizer, train_loader, EPOCHS)

######################################################################
# Evaluate the model with test dataset
# ------------------------------------


_, Y_actual, Y_preds = EvaluateModel(classifier, loss_fn, test_loader)

print("\nTest Accuracy : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))

