#!/usr/bin/env python3
"""
Evaluate saved LSTM text model on the validation set.

- Loads tokenizer from `outputs/tokenizer.pkl`
- Loads model from a given path (Keras .keras or SavedModel dir)
- Builds validation sequences from the validation JSONL (no retraining)
- Computes accuracy, confusion matrix, precision/recall/F1
- Prints a few wrong predictions with text snippets for inspection

Usage:
    python scripts/evaluate_text_model.py --model models/lstm_model.keras --tokenizer outputs/tokenizer.pkl --val data/fakeddit_subset/validation_data_fakeddit.jsonl --pad 128 --top 10

Simple and self-explanatory output suitable for quick debugging.
"""

from pathlib import Path
import argparse
import json
import re
import pickle
import numpy as np


def load_jsonl(path: Path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# Compatibility SimpleTokenizer so pickle.loads finds the class when file was saved from a script
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

    def texts_to_sequences(self, texts):
        # If tokenizer was not fitted in this process, rely on word_index attribute restored by pickle
        seqs = []
        for t in texts:
            if not hasattr(self, 'word_index'):
                seqs.append([])
                continue
            seqs.append([self.word_index.get(w, self.word_index['<UNK>']) for w in self.tokenize(t)])
        return seqs



def extract_text(record, exclude_model_answers: bool = False):
    texts = []

    def walk(o):
        if isinstance(o, str):
            # skip URIs and image filenames
            if '://' in o or 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)", o, re.I):
                return
            texts.append(o.replace('\n', ' '))
        elif isinstance(o, dict):
            # optionally skip 'model' role content
            if exclude_model_answers:
                role = (o.get('role') or '').lower() if isinstance(o.get('role'), str) else ''
                if role == 'model':
                    return
            for k, v in o.items():
                if k.lower() in ('text', 'title', 'caption') and isinstance(v, str):
                    texts.append(v.replace('\n', ' '))
                else:
                    walk(v)
        elif isinstance(o, list):
            for e in o:
                walk(e)

    walk(record)
    return ' '.join(t for t in texts if t).strip()


def extract_label(record):
    # Prefer explicit model role answers when present
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

    # fallback: keyword search
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
    joined = ' '.join(label_candidates).lower()

    if re.search(r"\bfake\b|\bhoax\b|\bphotoshop\b", joined):
        return 1
    if re.search(r"\breal\b|\btrue\b|\bgenuine\b", joined):
        return 0
    m2 = re.search(r"\blabel[:=]\s*(\d+)\b", joined)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            pass
    return -1


def pad_sequences(sequences, maxlen: int, padding: str = 'post'):
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if padding == 'post':
            out[i, :min(len(seq), maxlen)] = seq[:maxlen]
        else:
            out[i, max(0, maxlen - len(seq)):] = seq[-maxlen:]
    return out

def clean_text(text: str) -> str:
    if not text:
        return text
    import re
    m = re.search(r'Title:\s*"([^"]+)"', text, flags=re.I)
    if m:
        text = m.group(1)
    text = re.sub(r"\(Only Answer\s*\"?Yes\"?\/?\"?No\"?\)", "", text, flags=re.I)
    text = re.sub(r"Only Answer\s*\"?Yes\"?\/?\"?No\"?", "", text, flags=re.I)
    text = re.sub(r"\bis the picture[^\n\r]*\?", "", text, flags=re.I)
    text = re.sub(r"\bmodel\b\s*\w*", "", text, flags=re.I)
    text = re.sub(r"\b(yes|no)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_validation_set(val_json: Path, tokenizer, pad: int, exclude_model_answers: bool = False):
    records = load_jsonl(val_json)
    texts = []
    labels = []
    for rec in records:
        txt = extract_text(rec, exclude_model_answers=exclude_model_answers)
        txt = clean_text(txt)
        lab = extract_label(rec)
        if lab not in (0, 1):
            continue
        texts.append(txt)
        labels.append(lab)
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=pad, padding='post')
    y = np.array(labels, dtype=np.int32)
    return texts, X, y


def load_model(model_path: Path):
    try:
        import tensorflow as tf
        # try .keras file first
        if model_path.is_file() and model_path.suffix in ('.keras', '.h5'):
            model = tf.keras.models.load_model(str(model_path))
            return model
        elif model_path.is_dir():
            # SavedModel dir
            model = tf.keras.models.load_model(str(model_path))
            return model
        else:
            # try appending .keras
            p = model_path.with_suffix('.keras')
            if p.exists():
                return tf.keras.models.load_model(str(p))
    except Exception as e:
        raise RuntimeError('Failed to load TensorFlow/Keras model: ' + str(e))
    raise FileNotFoundError(f'Model not found at {model_path} (.keras or SavedModel dir expected)')


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved LSTM text model on validation set')
    parser.add_argument('--model', type=Path, required=True, help='Path to model (.keras or SavedModel dir)')
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/tokenizer.pkl'), help='Path to tokenizer.pkl')
    parser.add_argument('--val', type=Path, required=True, help='Path to validation JSONL')
    parser.add_argument('--pad', type=int, default=128, help='Pad length used during training')
    parser.add_argument('--top', type=int, default=10, help='Number of wrong predictions to show')
    parser.add_argument('--exclude-model-answers', action='store_true', help='Exclude model role answers from extracted text to avoid label leakage')
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f'Model not found: {args.model}')
    if not args.tokenizer.exists():
        raise FileNotFoundError(f'Tokenizer not found: {args.tokenizer}')

    with args.tokenizer.open('rb') as f:
        tokenizer = pickle.load(f)

    texts, X_val, y_val = build_validation_set(args.val, tokenizer, pad=args.pad, exclude_model_answers=args.exclude_model_answers)
    print(f'Validation items: {len(y_val)}')
    if len(y_val) == 0:
        raise RuntimeError('No labeled validation samples found. Check label extraction rules.')

    model = load_model(args.model)

    # Predict probabilities
    probs = model.predict(X_val, batch_size=64).ravel()
    preds = (probs >= 0.5).astype(int)

    # Metrics
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
        acc = accuracy_score(y_val, preds)
        cm = confusion_matrix(y_val, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='binary')
        print('\n=== Evaluation Summary ===')
        print(f'Accuracy: {acc:.4f}')
        print('Confusion matrix:\n', cm)
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')
        print('\nClassification report:\n')
        print(classification_report(y_val, preds, digits=4))
    except Exception:
        # minimal metrics if sklearn not available
        acc = (preds == y_val).mean()
        tp = int(((preds == 1) & (y_val == 1)).sum())
        tn = int(((preds == 0) & (y_val == 0)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print('\n=== Evaluation Summary ===')
        print(f'Accuracy: {acc:.4f}')
        print('Confusion matrix:')
        print(f'tp={tp}, tn={tn}, fp={fp}, fn={fn}')
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')

    # Show some wrong predictions
    wrong_idx = np.where(preds != y_val)[0]
    print(f'\nNumber of wrong predictions: {len(wrong_idx)}')
    top_n = min(args.top, len(wrong_idx))
    if top_n > 0:
        print(f'\n=== Examples of wrong predictions (up to {top_n}) ===')
        for i in wrong_idx[:top_n]:
            txt = texts[i]
            print('\n---')
            print(f'Index: {i}, True: {y_val[i]}, Pred: {preds[i]}, Prob: {probs[i]:.4f}')
            snippet = txt if len(txt) < 400 else txt[:400] + '...[truncated]'
            print(f'Text: "{snippet}"')


if __name__ == '__main__':
    main()
