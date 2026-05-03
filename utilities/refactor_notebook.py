import json

file_path = 'Part_C_RNNs copy.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

master_cell_source = [
    "# ============================================================\n",
    "# HELPER FUNCTIONS & MODEL DEFINITIONS\n",
    "# ============================================================\n",
    "def train_model(model, loss_fn, optimizer, train_loader, epochs):\n",
    "    epoch_times = []\n",
    "    for epoch in range(1, epochs + 1):\n",
    "        model.train()\n",
    "        losses = []\n",
    "        start = time.time()\n",
    "        for X, Y in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):\n",
    "            preds = model(X)\n",
    "            loss = loss_fn(preds, Y)\n",
    "            losses.append(loss.item())\n",
    "            optimizer.zero_grad()\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "        elapsed = time.time() - start\n",
    "        epoch_times.append(elapsed)\n",
    "        print(f'Epoch {epoch:2d} | Loss: {np.mean(losses):.4f} | Time: {elapsed:.1f}s')\n",
    "    return epoch_times\n",
    "\n",
    "def evaluate_model(model, loss_fn, test_loader):\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        Y_actual, Y_preds, losses = [], [], []\n",
    "        for X, Y in test_loader:\n",
    "            preds = model(X)\n",
    "            loss = loss_fn(preds, Y)\n",
    "            losses.append(loss.item())\n",
    "            Y_actual.append(Y)\n",
    "            Y_preds.append(preds.argmax(dim=-1))\n",
    "        Y_actual = torch.cat(Y_actual)\n",
    "        Y_preds = torch.cat(Y_preds)\n",
    "    return (torch.tensor(losses).mean().item(),\n",
    "            Y_actual.detach().cpu().numpy(),\n",
    "            Y_preds.detach().cpu().numpy())\n",
    "\n",
    "class RNNClassifier(nn.Module):\n",
    "    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,\n",
    "                 rnn_type='RNN', bidirectional=False, num_layers=1,\n",
    "                 pretrained_embeddings=None, freeze_embeddings=False):\n",
    "        super(RNNClassifier, self).__init__()\n",
    "        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)\n",
    "\n",
    "        if pretrained_embeddings is not None:\n",
    "            self.embedding_layer.weight.data.copy_(pretrained_embeddings)\n",
    "            if freeze_embeddings:\n",
    "                self.embedding_layer.weight.requires_grad = False\n",
    "\n",
    "        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.RNN\n",
    "        self.rnn = rnn_cls(input_size=embedding_dim, hidden_size=hidden_dim,\n",
    "                           num_layers=num_layers, batch_first=True,\n",
    "                           bidirectional=bidirectional)\n",
    "\n",
    "        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim\n",
    "        self.linear = nn.Linear(linear_input_dim, output_dim)\n",
    "        self.rnn_type = rnn_type\n",
    "\n",
    "    def forward(self, X_batch):\n",
    "        embeddings = self.embedding_layer(X_batch)\n",
    "        output, hidden = self.rnn(embeddings)\n",
    "        logits = self.linear(output[:, -1])\n",
    "        return F.softmax(logits, dim=1)\n",
    "\n",
    "def count_parameters(model):\n",
    "    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "\n",
    "architectures = [\n",
    "    {'name': '1RNN',       'rnn_type': 'RNN',  'bidirectional': False, 'num_layers': 1},\n",
    "    {'name': '1Bi-RNN',    'rnn_type': 'RNN',  'bidirectional': True,  'num_layers': 1},\n",
    "    {'name': '2Bi-RNN',    'rnn_type': 'RNN',  'bidirectional': True,  'num_layers': 2},\n",
    "    {'name': '1LSTM',      'rnn_type': 'LSTM', 'bidirectional': False, 'num_layers': 1},\n",
    "    {'name': '1Bi-LSTM',   'rnn_type': 'LSTM', 'bidirectional': True,  'num_layers': 1},\n",
    "    {'name': '2Bi-LSTM',   'rnn_type': 'LSTM', 'bidirectional': True,  'num_layers': 2},\n",
    "]\n",
    "\n",
    "# Load GloVe globally so it only downloads/parses once across all variations\n",
    "global_glove = None\n",
    "\n",
    "# ============================================================\n",
    "# MASTER EVALUATION LOOP\n",
    "# ============================================================\n",
    "def run_variation(var_name, max_words, use_glove, freeze_embeddings, use_imdb):\n",
    "    global global_glove\n",
    "    \n",
    "    print(f\"\\n{'#'*80}\")\n",
    "    print(f\"# RUNNING VARIATION: {var_name}\")\n",
    "    print(f\"{'#'*80}\\n\")\n",
    "    \n",
    "    tokenizer = get_tokenizer('basic_english')\n",
    "\n",
    "    # Data Loading\n",
    "    if use_imdb:\n",
    "        from torch.utils.data.dataset import random_split\n",
    "        imdb_data = pd.read_csv('IMDB Dataset.csv')\n",
    "        imdb_data['label'] = (imdb_data['sentiment'] == 'positive').astype(int)\n",
    "        all_dataset = [(row['label'], row['review'].lower()) for _, row in imdb_data.iterrows()]\n",
    "        train_size = int(0.8 * len(all_dataset))\n",
    "        test_size = len(all_dataset) - train_size\n",
    "        train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))\n",
    "        train_dataset = list(train_dataset)\n",
    "        test_dataset = list(test_dataset)\n",
    "        target_classes = ['Negative', 'Positive']\n",
    "    else:\n",
    "        train_data = pd.read_csv('ag-news-classification-dataset/train.csv')\n",
    "        test_data = pd.read_csv('ag-news-classification-dataset/test.csv')\n",
    "        train_dataset = [(label, train_data['Title'][i] + ' ' + train_data['Description'][i]) for i, label in enumerate(train_data['Class Index'])]\n",
    "        test_dataset = [(label, test_data['Title'][i] + ' ' + test_data['Description'][i]) for i, label in enumerate(test_data['Class Index'])]\n",
    "        target_classes = ['World', 'Sports', 'Business', 'Sci/Tech']\n",
    "        \n",
    "    num_classes = len(target_classes)\n",
    "    \n",
    "    # Vocab Building\n",
    "    def build_vocabulary(datasets):\n",
    "        for dataset in datasets:\n",
    "            for _, text in dataset:\n",
    "                yield tokenizer(text)\n",
    "\n",
    "    vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=10, specials=['<PAD>', '<UNK>'])\n",
    "    vocab.set_default_index(vocab['<UNK>'])\n",
    "    \n",
    "    label_offset = 0 if use_imdb else 1\n",
    "\n",
    "    def collate_batch(batch):\n",
    "        Y, X = list(zip(*batch))\n",
    "        Y = torch.tensor(Y) - label_offset\n",
    "        X = [vocab(tokenizer(text)) for text in X]\n",
    "        X = [tokens + ([vocab['<PAD>']] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in X]\n",
    "        return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device)\n",
    "\n",
    "    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)\n",
    "    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)\n",
    "    \n",
    "    # GloVe Init\n",
    "    pretrained_vectors = None\n",
    "    if use_glove:\n",
    "        print('Loading GloVe 6B-100d...')\n",
    "        if global_glove is None:\n",
    "            global_glove = api.load('glove-wiki-gigaword-100')\n",
    "        pretrained_vectors = torch.zeros(len(vocab), EMBEDDING_DIM)\n",
    "        found = 0\n",
    "        for idx, word in enumerate(vocab.get_itos()):\n",
    "            if word in global_glove:\n",
    "                pretrained_vectors[idx] = torch.tensor(global_glove[word], dtype=torch.float32)\n",
    "                found += 1\n",
    "        print(f'GloVe vectors loaded. {found}/{len(vocab)} words found.')\n",
    "\n",
    "    all_results = {}\n",
    "    trained_models = {}\n",
    "    \n",
    "    for arch in architectures:\n",
    "        arch_name = arch['name']\n",
    "        print(f\"\\n{'='*50}\")\n",
    "        print(f\"  Architecture: {arch_name}\")\n",
    "        print(f\"{'='*50}\")\n",
    "        run_accs, run_times = [], []\n",
    "        n_params = None\n",
    "\n",
    "        for run_idx in range(1, NUM_RUNS + 1):\n",
    "            print(f\"\\n  --- Run {run_idx}/{NUM_RUNS} ---\")\n",
    "            model = RNNClassifier(\n",
    "                vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,\n",
    "                output_dim=num_classes, rnn_type=arch['rnn_type'], bidirectional=arch['bidirectional'],\n",
    "                num_layers=arch['num_layers'], pretrained_embeddings=pretrained_vectors, freeze_embeddings=freeze_embeddings\n",
    "            ).to(device)\n",
    "\n",
    "            if n_params is None:\n",
    "                n_params = count_parameters(model)\n",
    "                print(f\"  Parameters: {n_params:,}\")\n",
    "                \n",
    "            loss_fn = nn.CrossEntropyLoss()\n",
    "            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)\n",
    "\n",
    "            epoch_times = train_model(model, loss_fn, optimizer, train_loader, EPOCHS)\n",
    "            _, Y_actual, Y_preds = evaluate_model(model, loss_fn, test_loader)\n",
    "            acc = accuracy_score(Y_actual, Y_preds)\n",
    "\n",
    "            run_accs.append(acc)\n",
    "            run_times.append(np.mean(epoch_times))\n",
    "            print(f'  Run {run_idx} Test Accuracy: {acc:.4f}')\n",
    "            trained_models[arch_name] = model\n",
    "            \n",
    "        all_results[arch_name] = {\n",
    "            'mean_acc': np.mean(run_accs), 'std_acc': np.std(run_accs),\n",
    "            'params': n_params, 'mean_time_per_epoch': np.mean(run_times)\n",
    "        }\n",
    "        \n",
    "    print(f\"\\n{'Architecture':<15s} {'Mean Acc':>10s} {'Std Acc':>10s} {'Params':>12s} {'Time/Epoch':>12s}\")\n",
    "    print('-' * 62)\n",
    "    for name, r in all_results.items():\n",
    "        print(f\"{name:<15s} {r['mean_acc']:>10.4f} {r['std_acc']:>10.4f} {r['params']:>12,} {r['mean_time_per_epoch']:>10.2f}s\")\n",
    "        \n",
    "    return all_results, trained_models, vocab\n",
    "\n",
    "# ============================================================\n",
    "# SEQUENTIALLY RUN ALL CONFIGURATIONS\n",
    "# ============================================================\n",
    "variations = [\n",
    "    {\"var_name\": \"Base (AG News, 25 words)\", \"max_words\": 25, \"use_glove\": False, \"freeze_embeddings\": False, \"use_imdb\": False},\n",
    "    {\"var_name\": \"50 Max Words (AG News)\", \"max_words\": 50, \"use_glove\": False, \"freeze_embeddings\": False, \"use_imdb\": False},\n",
    "    {\"var_name\": \"GloVe Trainable (AG News, 25 words)\", \"max_words\": 25, \"use_glove\": True, \"freeze_embeddings\": False, \"use_imdb\": False},\n",
    "    {\"var_name\": \"GloVe Frozen (AG News, 25 words)\", \"max_words\": 25, \"use_glove\": True, \"freeze_embeddings\": True, \"use_imdb\": False},\n",
    "    {\"var_name\": \"IMDB Dataset (25 words)\", \"max_words\": 25, \"use_glove\": False, \"freeze_embeddings\": False, \"use_imdb\": True},\n",
    "]\n",
    "\n",
    "master_results = {}\n",
    "base_rnn1_model = None\n",
    "base_vocab = None\n",
    "\n",
    "for var in variations:\n",
    "    res, models, var_vocab = run_variation(**var)\n",
    "    master_results[var['var_name']] = res\n",
    "    \n",
    "    # Save the 1RNN model and vocab from the Base variation for the t-SNE plot later\n",
    "    if var['var_name'] == \"Base (AG News, 25 words)\":\n",
    "        base_rnn1_model = models.get('1RNN')\n",
    "        base_vocab = var_vocab\n",
    "\n",
    "print('\\n\\nALL EXPERIMENTS COMPLETE!')\n"
]

new_cells = []
skip = False

for cell in nb['cells']:
    source = cell.get('source', [])
    source_str = "".join(source) if isinstance(source, list) else source
    
    if "## 2. Hyperparameters" in source_str:
        new_cells.append(cell)
        
        hyper_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Static Hyperparameters (These apply to all variations)\n",
                "EPOCHS = 15\n",
                "LEARNING_RATE = 1e-3\n",
                "BATCH_SIZE = 1024\n",
                "EMBEDDING_DIM = 100\n",
                "HIDDEN_DIM = 64\n",
                "NUM_RUNS = 3\n"
            ]
        }
        new_cells.append(hyper_cell)
        
        master_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": master_cell_source
        }
        new_cells.append(master_cell)
        
        skip = True
        continue
        
    if skip:
        if "## 11. t-SNE" in source_str:
            skip = False
        else:
            continue
            
    if not skip:
        if "rnn1_model = trained_models.get('1RNN')" in source_str:
            new_source = []
            for line in source:
                patched = line.replace("rnn1_model = trained_models.get('1RNN')", "rnn1_model = base_rnn1_model")
                patched = patched.replace("idx = vocab[word]", "idx = base_vocab[word]")
                patched = patched.replace("idx != vocab['<UNK>']", "idx != base_vocab['<UNK>']")
                new_source.append(patched)
            cell['source'] = new_source
        new_cells.append(cell)

nb['cells'] = new_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Refactoring complete.')
