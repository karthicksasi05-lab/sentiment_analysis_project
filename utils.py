"""Helper functions shared by app.py and predict.py.

Everything here is copied from the notebook so the app cleans and encodes
text in exactly the same way as during training.

The RNN / LSTM / GRU model is always available after running the main notebook.
FinBERT is optional: it only appears if you trained it with FinBERT_Colab.ipynb
and copied the models/finbert_sentiment folder here.
"""
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
PAD = "<pad>"
UNK = "<unk>"

MODEL_FOLDER = "models"
BERT_FOLDER = os.path.join(MODEL_FOLDER, "finbert_sentiment")
BERT_RESULT_FILE = os.path.join("data", "bert_results.json")
COMPARISON_FILE = os.path.join("data", "model_comparison.csv")


# ---------------------------------------------------------------- text steps
def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # links
    text = re.sub(r"@\w+", " ", text)                    # mentions
    text = re.sub(r"<[^>]+>", " ", text)                 # html tags
    text = re.sub(r"#MarketScreener\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return re.findall(r"\$?[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+", text.lower())


def text_to_ids(text, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK]) for t in tokenize(text)]
    ids = ids[:max_len]
    ids = ids + [vocab[PAD]] * (max_len - len(ids))
    return ids


def split_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p.strip() != ""]


# ---------------------------------------------------------------- baseline model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=3, rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last = self.dropout(hidden[-1])
        return self.fc(last)


def baseline_available():
    needed = ["best_baseline_model.pt", "model_config.json", "vocab.json"]
    return all(os.path.exists(os.path.join(MODEL_FOLDER, f)) for f in needed)


def load_baseline():
    """Returns (model, vocab, config) for the saved RNN/LSTM/GRU model."""
    with open(os.path.join(MODEL_FOLDER, "model_config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_FOLDER, "vocab.json")) as f:
        vocab = json.load(f)

    model = SentimentModel(config["vocab_size"], config["embed_dim"], config["hidden_dim"],
                           config["num_classes"], config["rnn_type"])
    state = torch.load(os.path.join(MODEL_FOLDER, "best_baseline_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, vocab, config


def baseline_probs(sentence, model, vocab, max_len):
    ids = text_to_ids(clean_text(sentence), vocab, max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()


# ---------------------------------------------------------------- optional FinBERT model
def bert_available():
    return os.path.isdir(BERT_FOLDER)


def load_bert():
    """Returns (model, tokenizer) for the fine-tuned FinBERT (needs the transformers library)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_FOLDER)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_FOLDER)
    model.eval()
    return model, tokenizer


def bert_probs(sentence, model, tokenizer):
    inputs = tokenizer(clean_text(sentence), return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0].numpy()


# ---------------------------------------------------------------- comparison table
def load_comparison():
    """Table of model scores. Adds the FinBERT row if data/bert_results.json exists."""
    import pandas as pd

    if not os.path.exists(COMPARISON_FILE):
        return None
    table = pd.read_csv(COMPARISON_FILE)

    if os.path.exists(BERT_RESULT_FILE):
        with open(BERT_RESULT_FILE) as f:
            bert_row = json.load(f)
        table = table[~table["Model"].str.contains("BERT")]
        table = pd.concat([table, pd.DataFrame([bert_row])], ignore_index=True)

    return table.sort_values("Val Macro F1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- predicting a paragraph
def predict_paragraph(paragraph, probs_function):
    """Predict a whole paragraph.

    probs_function takes one sentence and returns 3 probabilities.
    We predict every sentence, average the probabilities and take the biggest one.
    Returns (label, average probabilities dict, list of (sentence, label)).
    """
    sentences = split_sentences(paragraph)
    if len(sentences) == 0:
        return None, None, []

    all_probs = []
    per_sentence = []
    for s in sentences:
        p = probs_function(s)
        all_probs.append(p)
        per_sentence.append((s, LABEL_MAP[int(np.argmax(p))]))

    avg = np.mean(all_probs, axis=0)
    label = LABEL_MAP[int(np.argmax(avg))]
    probs = {LABEL_MAP[i]: float(avg[i]) for i in range(3)}
    return label, probs, per_sentence
