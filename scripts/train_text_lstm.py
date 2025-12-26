#!/usr/bin/env python3
"""
Train a simple LSTM text classifier (fake vs real) using prepared tokenizer
and JSONL split files (training and validation). The script:
- Loads tokenizer (from preprocessing outputs)
- Extracts text and labels from train/validation JSONL
- Converts text to token id sequences and pads to fixed length
- Builds a simple Keras model: Embedding -> LSTM -> Dropout -> Sigmoid
- Trains and prints training/validation loss and accuracy per epoch
- Saves the trained model to `models/lstm_model` (SavedModel format)

Usage:
    python scripts/train_text_lstm.py --train data/fakeddit_subset/training_data_fakeddit.jsonl \
        --val data/fakeddit_subset/validation_data_fakeddit.jsonl --tokenizer outputs/tokenizer.pkl --out models

"""

from pathlib import Path
import argparse
import json
import re
import pickle
from collections import Counter
from typing import List, Tuple, Any
import numpy as np

# --- TensorFlow imports will be lazy to allow installation if missing ---


# --- Helpers (similar to preprocessing) ---

# Compatibility SimpleTokenizer (same structure as used during preprocessing save)
class SimpleTokenizer:
    def __init__(self, lower=True, token_pattern=r"\b\w+\b"):
        import re
        from collections import Counter
        self.lower = lower
        self.token_pattern = re.compile(token_pattern)
        self.word_counts = Counter()
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_word = {0: "<PAD>", 1: "<UNK>"}
        self.fitted = False

    def tokenize(self, text: str):
        if self.lower:
            text = text.lower()
        return self.token_pattern.findall(text)

    def fit_on_texts(self, texts, min_count: int = 1, max_vocab: int = None):
        for t in texts:
            self.word_counts.update(self.tokenize(t))
        items = [(w, c) for w, c in self.word_counts.items() if c >= min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab:
            items = items[:max_vocab]
        idx = 2
        for w, _ in items:
            if w not in self.word_index:
                self.word_index[w] = idx
                self.index_word[idx] = w
                idx += 1
        self.fitted = True

    def texts_to_sequences(self, texts):
        if not self.fitted:
            # allow padding: if not fitted, treat all tokens as UNK
            return [[self.word_index['<UNK>'] for _ in (text.split() if text else [])] for text in texts]
        seqs = []
        for t in texts:
            seq = [self.word_index.get(w, self.word_index['<UNK>']) for w in self.tokenize(t)]
            seqs.append(seq)
        return seqs

    def save(self, path: Path):
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with path.open('rb') as f:
            return pickle.load(f)


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse JSON on line {i}: {e}")
    return records


def extract_text(record: Any, exclude_model_answers: bool = False) -> str:
    texts = []

    def walk(o):
        if isinstance(o, str):
            if '://' in o or 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)", o, re.I):
                return
            texts.append(o.replace("\n", " "))
        elif isinstance(o, dict):
            # optionally skip 'model' role dicts
            if exclude_model_answers:
                role = (o.get('role') or '').lower() if isinstance(o.get('role'), str) else ''
                if role == 'model':
                    return
            for k, v in o.items():
                if k.lower() in ('text', 'title', 'caption') and isinstance(v, str):
                    texts.append(v.replace("\n", " "))
                else:
                    walk(v)
        elif isinstance(o, list):
            for e in o:
                walk(e)

    walk(record)
    return " ".join(t for t in texts if t).strip()


def extract_label(record: Any) -> int:
    # Prefer model role answer when present
    def find_model_answer(o):
        if isinstance(o, dict):
            role = (o.get('role') or '').lower()
            if role == 'model':
                parts = o.get('parts', [])
                for p in parts:
                    if isinstance(p, dict):
                        txt = p.get('text')
                        if isinstance(txt, str):
                            t = txt.strip().lower()
                            if t in ('yes', 'true', '1', 'fake', 'hoax'):
                                return 1
                            if t in ('no', 'false', '0', 'real', 'genuine', 'authentic'):
                                return 0
            for v in o.values():
                res = find_model_answer(v)
                if res is not None:
                    return res
        elif isinstance(o, list):
            for e in o:
                res = find_model_answer(e)
                if res is not None:
                    return res
        return None

    m = find_model_answer(record)
    if m is not None:
        return m

    # fallback to keyword search
    label_candidates = []
    def walk(o):
        if isinstance(o, str):
            label_candidates.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for e in o:
                walk(e)
    walk(record)
    joined = " ".join(label_candidates).lower()
    if re.search(r"\bfake\b|\bhoax\b|\bphotoshop\b", joined):
        return 1
    if re.search(r"\breal\b|\btrue\b|\bgenuine\b", joined):
        return 0
    m = re.search(r"\blabel[:=]\s*(\d+)\b", joined)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return -1


def pad_sequences(sequences: List[List[int]], maxlen: int, padding: str = 'post') -> np.ndarray:
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if padding == 'post':
            out[i, :min(len(seq), maxlen)] = seq[:maxlen]
        else:
            out[i, max(0, maxlen - len(seq)):] = seq[-maxlen:]
    return out


def build_dataset_from_jsonl(path: Path, tokenizer, maxlen: int, exclude_model_answers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    records = load_jsonl(path)
    texts = []
    labels = []
    for rec in records:
        txt = extract_text(rec, exclude_model_answers=exclude_model_answers)
        lab = extract_label(rec)
        if lab not in (0, 1):
            continue  # skip unknown labels
        texts.append(txt)
        labels.append(lab)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=maxlen, padding='post')
    return padded, np.array(labels, dtype=np.float32)


def build_model(vocab_size: int, embed_dim: int, lstm_units: int, dropout: float, input_length: int):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length),
        LSTM(lstm_units),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train LSTM text classifier')
    parser.add_argument('--train', type=Path, required=True, help='Path to training JSONL')
    parser.add_argument('--val', type=Path, required=True, help='Path to validation JSONL')
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/tokenizer.pkl'), help='Path to tokenizer.pkl')
    parser.add_argument('--out', type=Path, default=Path('models'), help='Model output dir')
    parser.add_argument('--pad', type=int, default=128, help='Pad length')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--lstm', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exclude-model-answers', action='store_true', help='Exclude model role answers from extracted text to avoid label leakage')
    args = parser.parse_args()

    # load tokenizer
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    with args.tokenizer.open('rb') as f:
        tokenizer = pickle.load(f)

    # build datasets separately (DO NOT mix train/val)
    print('Building training dataset...')
    X_train, y_train = build_dataset_from_jsonl(args.train, tokenizer, maxlen=args.pad, exclude_model_answers=args.exclude_model_answers)
    print(f'Train samples: {len(X_train)}')

    print('Building validation dataset...')
    X_val, y_val = build_dataset_from_jsonl(args.val, tokenizer, maxlen=args.pad, exclude_model_answers=args.exclude_model_answers)
    print(f'Validation samples: {len(X_val)}')

    # quick sanity checks
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError('Empty training or validation set after processing labels.')

    # model
    print('Building model...')
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError('TensorFlow is required. Please install it in the Python environment.')

    model = build_model(vocab_size=len(tokenizer.word_index), embed_dim=args.embed, lstm_units=args.lstm, dropout=args.dropout, input_length=args.pad)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # train and observe metrics
    print('Training...')
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=args.batch, epochs=args.epochs)

    # Print observed metrics per epoch
    print('\nTraining history:')
    for epoch in range(len(history.history['loss'])):
        t_loss = history.history['loss'][epoch]
        v_loss = history.history['val_loss'][epoch]
        t_acc = history.history['accuracy'][epoch]
        v_acc = history.history['val_accuracy'][epoch]
        print(f'Epoch {epoch+1}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, train_acc={t_acc:.4f}, val_acc={v_acc:.4f}')

    # save model
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras native format (.keras) and also export SavedModel for TF serving if possible
    keras_path = out_dir / 'lstm_model.keras'
    model.save(keras_path)
    print(f'Model saved (Keras format) to {keras_path}')

    saved_dir = out_dir / 'lstm_model'
    try:
        # Keras 3: export is the recommended way to get a SavedModel
        model.export(saved_dir)
        print(f'Model exported (SavedModel) to {saved_dir}')
    except Exception:
        # fallback to tf.saved_model
        import tensorflow as _tf
        _tf.saved_model.save(model, str(saved_dir))
        print(f'Model saved via tf.saved_model.save to {saved_dir}')


if __name__ == '__main__':
    main()
