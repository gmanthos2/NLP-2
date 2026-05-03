%pip install torch==2.3.0 torchtext==0.18.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall
%pip install pandas numpy tqdm scikit-learn matplotlib seaborn gensim
---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np
import time
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api

sns.set_style('whitegrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
---
# ============================================================
# TOGGLEABLE CONFIGURATION — change these to run variations
# ============================================================
MAX_WORDS = 25            # Set to 50 for the MAX_WORDS variation
EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
NUM_RUNS = 3              # Number of repeated runs per architecture

USE_GLOVE = False         # Set True to init embeddings with glove-6B-100d
FREEZE_EMBEDDINGS = False # Set True to freeze pre-trained embeddings
USE_IMDB = False          # Set True to use IMDB dataset instead of AG News

print(f'MAX_WORDS={MAX_WORDS}, USE_GLOVE={USE_GLOVE}, FREEZE={FREEZE_EMBEDDINGS}, USE_IMDB={USE_IMDB}')
---
tokenizer = get_tokenizer('basic_english')

if USE_IMDB:
    # IMDB dataset (2 classes)
    from torch.utils.data.dataset import random_split
    imdb_data = pd.read_csv('IMDB Dataset.csv')  # expects 'review' and 'sentiment' columns
    # If file not found, try downloading:
    # imdb_data = pd.read_csv('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz') 
    # For simplicity, we assume the CSV is available
    imdb_data['label'] = (imdb_data['sentiment'] == 'positive').astype(int)
    all_dataset = [(row['label'], row['review'].lower()) for _, row in imdb_data.iterrows()]
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    target_classes = ['Negative', 'Positive']
    print(f'IMDB dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test')
else:
    # AG News dataset (4 classes)
    train_data = pd.read_csv('ag-news-classification-dataset/train.csv')
    test_data = pd.read_csv('ag-news-classification-dataset/test.csv')
    train_dataset = [(label, train_data['Title'][i] + ' ' + train_data['Description'][i])
                     for i, label in enumerate(train_data['Class Index'])]
    test_dataset = [(label, test_data['Title'][i] + ' ' + test_data['Description'][i])
                    for i, label in enumerate(test_data['Class Index'])]
    target_classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    print(f'AG News dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test')

NUM_CLASSES = len(target_classes)
print(f'Classes ({NUM_CLASSES}): {target_classes}')
---
def build_vocabulary(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

vocab = build_vocab_from_iterator(
    build_vocabulary([train_dataset, test_dataset]),
    min_freq=10,
    specials=['<PAD>', '<UNK>']
)
vocab.set_default_index(vocab['<UNK>'])
print(f'Vocabulary size: {len(vocab)}')

# Label offset: AG News labels are 1-indexed, IMDB are already 0-indexed
LABEL_OFFSET = 0 if USE_IMDB else 1

def collate_batch(batch):
    Y, X = list(zip(*batch))
    Y = torch.tensor(Y) - LABEL_OFFSET
    X = [vocab(tokenizer(text)) for text in X]
    X = [tokens + ([vocab['<PAD>']] * (MAX_WORDS - len(tokens))) if len(tokens) < MAX_WORDS
         else tokens[:MAX_WORDS] for tokens in X]
    return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 rnn_type='RNN', bidirectional=False, num_layers=1,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Optionally load pre-trained embeddings
        if pretrained_embeddings is not None:
            self.embedding_layer.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding_layer.weight.requires_grad = False

        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.RNN
        self.rnn = rnn_cls(input_size=embedding_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional)

        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.linear = nn.Linear(linear_input_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings)
        # Use the last time-step output for classification
        logits = self.linear(output[:, -1])
        probs = F.softmax(logits, dim=1)
        return probs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Model class defined.')
---
pretrained_vectors = None

if USE_GLOVE:
    print('Loading GloVe 6B-100d...')
    glove = api.load('glove-wiki-gigaword-100')
    pretrained_vectors = torch.zeros(len(vocab), EMBEDDING_DIM)
    found = 0
    for idx, word in enumerate(vocab.get_itos()):
        if word in glove:
            pretrained_vectors[idx] = torch.tensor(glove[word], dtype=torch.float32)
            found += 1
    print(f'GloVe vectors loaded. {found}/{len(vocab)} words found.')
else:
    print('Skipping GloVe (USE_GLOVE=False)')
---
def train_model(model, loss_fn, optimizer, train_loader, epochs):
    epoch_times = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        start = time.time()
        for X, Y in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start
        epoch_times.append(elapsed)
        print(f'Epoch {epoch:2d} | Loss: {np.mean(losses):.4f} | Time: {elapsed:.1f}s')
    return epoch_times


def evaluate_model(model, loss_fn, test_loader):
    model.eval()
    with torch.no_grad():
        Y_actual, Y_preds, losses = [], [], []
        for X, Y in test_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            Y_actual.append(Y)
            Y_preds.append(preds.argmax(dim=-1))
        Y_actual = torch.cat(Y_actual)
        Y_preds = torch.cat(Y_preds)
    return (torch.tensor(losses).mean().item(),
            Y_actual.detach().cpu().numpy(),
            Y_preds.detach().cpu().numpy())
---
architectures = [
    {'name': '1RNN',       'rnn_type': 'RNN',  'bidirectional': False, 'num_layers': 1},
    {'name': '1Bi-RNN',    'rnn_type': 'RNN',  'bidirectional': True,  'num_layers': 1},
    {'name': '2Bi-RNN',    'rnn_type': 'RNN',  'bidirectional': True,  'num_layers': 2},
    {'name': '1LSTM',      'rnn_type': 'LSTM', 'bidirectional': False, 'num_layers': 1},
    {'name': '1Bi-LSTM',   'rnn_type': 'LSTM', 'bidirectional': True,  'num_layers': 1},
    {'name': '2Bi-LSTM',   'rnn_type': 'LSTM', 'bidirectional': True,  'num_layers': 2},
]

print('Architectures to evaluate:')
for a in architectures:
    print(f"  - {a['name']}")
---
all_results = {}
trained_models = {}  # Store last trained model per architecture

for arch in architectures:
    arch_name = arch['name']
    print(f"\n{'='*70}")
    print(f"  Architecture: {arch_name}")
    print(f"{'='*70}")

    run_accs = []
    run_times = []
    n_params = None

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n  --- Run {run_idx}/{NUM_RUNS} ---")

        model = RNNClassifier(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES,
            rnn_type=arch['rnn_type'],
            bidirectional=arch['bidirectional'],
            num_layers=arch['num_layers'],
            pretrained_embeddings=pretrained_vectors,
            freeze_embeddings=FREEZE_EMBEDDINGS
        ).to(device)

        if n_params is None:
            n_params = count_parameters(model)
            print(f'  Parameters: {n_params:,}')

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE
        )

        epoch_times = train_model(model, loss_fn, optimizer, train_loader, EPOCHS)
        _, Y_actual, Y_preds = evaluate_model(model, loss_fn, test_loader)
        acc = accuracy_score(Y_actual, Y_preds)

        run_accs.append(acc)
        run_times.append(np.mean(epoch_times))
        print(f'  Run {run_idx} Test Accuracy: {acc:.4f}')

        trained_models[arch_name] = model  # keep last run

    all_results[arch_name] = {
        'mean_acc': np.mean(run_accs),
        'std_acc': np.std(run_accs),
        'params': n_params,
        'mean_time_per_epoch': np.mean(run_times),
    }

print('\n\nAll runs complete!')
---
print(f"{'Architecture':<15s} {'Mean Acc':>10s} {'Std Acc':>10s} {'Params':>12s} {'Time/Epoch':>12s}")
print('-' * 62)
for name, r in all_results.items():
    print(f"{name:<15s} {r['mean_acc']:>10.4f} {r['std_acc']:>10.4f} {r['params']:>12,} {r['mean_time_per_epoch']:>10.2f}s")
---
tsne_words = [
    'business', 'career', 'student', 'university', 'college',
    'education', 'teacher', 'professor', 'school', 'degree',
    'economy', 'market', 'finance', 'investment', 'bank',
    'company', 'startup', 'entrepreneur', 'manager', 'salary',
    'science', 'research', 'technology', 'engineering', 'mathematics',
    'doctor', 'lawyer', 'accountant'
]

# Extract embeddings from the trained 1RNN model
rnn1_model = trained_models.get('1RNN')
if rnn1_model is None:
    print('ERROR: 1RNN model not found in trained_models dict.')
else:
    embedding_weights = rnn1_model.embedding_layer.weight.data.cpu().numpy()
    word_vectors = []
    valid_words = []
    for word in tsne_words:
        idx = vocab[word]  # returns <UNK> index if not found
        if idx != vocab['<UNK>']:
            word_vectors.append(embedding_weights[idx])
            valid_words.append(word)
        else:
            print(f"'{word}' not in vocabulary, skipping.")

    word_vectors = np.array(word_vectors)
    print(f'Collected {len(valid_words)} word vectors')

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=8, n_iter=2000)
    embeddings_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c='coral', s=100, alpha=0.7, edgecolors='darkred', linewidths=0.5)
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                     xytext=(7, 4), textcoords='offset points',
                     fontsize=11, fontweight='bold', color='darkslategray')

    plt.title('t-SNE of Learned 1RNN Embeddings (AG News)', fontsize=15, fontweight='bold')
    plt.xlabel('t-SNE Dim 1', fontsize=12)
    plt.ylabel('t-SNE Dim 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_rnn_part_c.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'tsne_rnn_part_c.png'")
