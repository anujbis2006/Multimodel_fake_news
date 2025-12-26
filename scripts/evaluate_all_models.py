#!/usr/bin/env python3
"""
Compare Text-only (LSTM), Image-only (CNN), and Multimodal models on the validation set.
- Loads tokenizer, models, validation JSONL and images
- Computes accuracy, confusion matrix, precision/recall/F1 per model
- Creates a comparison table (Model, Accuracy, Strength, Weakness)
- Picks 5 error cases where text failed and 5 where image failed and checks whether multimodal corrected them
- Prints short automatic observations for each case

Usage:
    python scripts/evaluate_all_models.py --val data/fakeddit_subset/validation_data_fakeddit.jsonl \
        --img-dir data/fakeddit_subset/validation_image --tokenizer outputs/tokenizer.pkl --models models

"""
from pathlib import Path
import argparse
import json
import re
import pickle
from typing import List, Tuple, Any, Dict
import numpy as np

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report


# Compatibility tokenizer class for unpickling
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
        seqs = []
        for t in texts:
            seqs.append([self.word_index.get(w, self.word_index['<UNK>']) for w in self.tokenize(t)])
        return seqs


# Extraction helpers (same rules as before)

def load_jsonl(path: Path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_text(record: Any, exclude_model_answers: bool = False) -> str:
    texts = []

    def walk(o):
        if isinstance(o, str):
            if '://' in o or 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)", o, re.I):
                return
            texts.append(o.replace('\n', ' '))
        elif isinstance(o, dict):
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


def extract_image_id(record: Any) -> str:
    found = None

    def walk(o):
        nonlocal found
        if found:
            return
        if isinstance(o, str):
            if 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)($|\?)", o, re.I) or o.startswith('http'):
                found = Path(o.split('://')[-1].split('/')[-1].split('?')[0]).name
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
                if found:
                    return
        elif isinstance(o, list):
            for e in o:
                walk(e)
                if found:
                    return

    walk(record)
    return found or ''


def extract_label(record: Any) -> int:
    # Prefer model role answers when present
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
    if re.search(r"\breal\b|\btrue\b|\bgenuine\b|\bauthentic\b", joined):
        return 0
    m = re.search(r"\blabel[:=]\s*(\d+)\b", joined)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return -1


def pad_sequences(seqs: List[List[int]], maxlen: int) -> np.ndarray:
    out = np.zeros((len(seqs), maxlen), dtype=np.int32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        out[i, :min(len(s), maxlen)] = s[:maxlen]
    return out


def clean_text(text: str) -> str:
    if not text:
        return text
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


def build_validation_lists(val_json: Path, img_dir: Path, tokenizer, pad: int, exclude_model_answers: bool = False):
    records = load_jsonl(val_json)
    texts = []
    img_paths = []
    labels = []
    raw_texts = []
    for rec in records:
        txt = extract_text(rec, exclude_model_answers=exclude_model_answers)
        txt = clean_text(txt)
        img_id = extract_image_id(rec)
        lab = extract_label(rec)
        if lab not in (0, 1):
            continue
        if not img_id:
            continue
        img_path = img_dir / img_id
        if not img_path.exists():
            continue
        texts.append(txt)
        raw_texts.append(txt)
        img_paths.append(str(img_path))
        labels.append(lab)
    seqs = tokenizer.texts_to_sequences(texts)
    X_text = pad_sequences(seqs, maxlen=pad)
    y = np.array(labels, dtype=np.int32)
    return texts, raw_texts, X_text, img_paths, y


def load_keras_model(mpath: Path):
    if not mpath.exists():
        # try adding .keras
        if mpath.with_suffix('.keras').exists():
            mpath = mpath.with_suffix('.keras')
        else:
            raise FileNotFoundError(f'Model not found: {mpath}')
    model = tf.keras.models.load_model(str(mpath))
    return model


def evaluate_preds(y_true: np.ndarray, probs: np.ndarray, thresh=0.5) -> Dict[str, Any]:
    preds = (probs >= thresh).astype(int)
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
    return {'preds': preds, 'probs': probs, 'acc': acc, 'cm': cm, 'prec': prec, 'rec': rec, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--img-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/tokenizer.pkl'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--pad', type=int, default=128)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--exclude-model-answers', action='store_true', help='Exclude model role answers from extracted text to avoid label leakage')
    args = parser.parse_args()

    with args.tokenizer.open('rb') as f:
        tokenizer = pickle.load(f)

    texts, raw_texts, X_text, img_paths, y = build_validation_lists(args.val, args.img_dir, tokenizer, pad=args.pad, exclude_model_answers=args.exclude_model_answers)
    print(f'Validation items: {len(y)}')

    # Load models
    text_model = load_keras_model(args.models / 'lstm_model.keras')
    image_model = load_keras_model(args.models / 'cnn_model.keras')
    multimodal_model = load_keras_model(args.models / 'multimodal_model.keras')

    # Text-only predictions
    text_probs = text_model.predict(X_text, batch_size=64).ravel()
    text_res = evaluate_preds(y, text_probs)

    # Image-only predictions: build dataset of images only (no shuffle)
    def img_ds_from_paths(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        def _prep(p):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (128, 128))
            img = img / 255.0
            return img
        ds = ds.map(_prep).batch(64)
        return ds

    img_ds = img_ds_from_paths(img_paths)
    image_probs = image_model.predict(img_ds).ravel()
    image_res = evaluate_preds(y, image_probs)

    # Multimodal predictions: create dataset yielding ({text_input:..., image_input:...}, label)
    def multimodal_ds(texts_arr, img_paths_arr):
        ds = tf.data.Dataset.from_tensor_slices((texts_arr, img_paths_arr))
        def _map(t, p):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (128, 128))
            img = img / 255.0
            return ({'text_input': t, 'image_input': img})
        ds = ds.map(_map).batch(64)
        return ds

    multimodal_input_ds = multimodal_ds(X_text, img_paths)
    multimodal_probs = multimodal_model.predict(multimodal_input_ds).ravel()
    multimodal_res = evaluate_preds(y, multimodal_probs)

    # Print metrics table
    def fmt_pct(x):
        return f"{x*100:.2f}%"

    print('\n=== Summary Metrics ===')
    header = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Strength', 'Weakness']
    rows = []
    rows.append(['Text-only', fmt_pct(text_res['acc']), f"{text_res['prec']:.3f}", f"{text_res['rec']:.3f}", f"{text_res['f1']:.3f}", 'Captures intent', 'Misses visual context'])
    rows.append(['Image-only', fmt_pct(image_res['acc']), f"{image_res['prec']:.3f}", f"{image_res['rec']:.3f}", f"{image_res['f1']:.3f}", 'Visual cues', 'Weak standalone'])
    rows.append(['Multimodal', fmt_pct(multimodal_res['acc']), f"{multimodal_res['prec']:.3f}", f"{multimodal_res['rec']:.3f}", f"{multimodal_res['f1']:.3f}", 'Balanced', 'More complex'])

    # Print table
    print('\nModel	Accuracy	Precision	Recall	F1	Strength	Weakness')
    for r in rows:
        print('\t'.join(r))

    # Detailed metrics prints
    print('\nDetailed metrics:')
    print('Text-only:')
    print(text_res['cm'])
    print(classification_report(y, text_res['preds'], digits=4))

    print('Image-only:')
    print(image_res['cm'])
    print(classification_report(y, image_res['preds'], digits=4))

    print('Multimodal:')
    print(multimodal_res['cm'])
    print(classification_report(y, multimodal_res['preds'], digits=4))

    # Pick error cases
    text_wrong_idx = np.where(text_res['preds'] != y)[0].tolist()
    image_wrong_idx = np.where(image_res['preds'] != y)[0].tolist()

    text_examples = text_wrong_idx[:args.top]
    image_examples = [i for i in image_wrong_idx if i not in text_examples][:args.top]

    print(f"\nPicked {len(text_examples)} text-error examples and {len(image_examples)} image-error examples (up to {args.top})")

    def describe_case(i):
        return {
            'index': i,
            'image': Path(img_paths[i]).name,
            'text': raw_texts[i][:400] + ('...[truncated]' if len(raw_texts[i]) > 400 else ''),
            'true': int(y[i]),
            'text_pred': int(text_res['preds'][i]),
            'text_prob': float(text_res['probs'][i]),
            'image_pred': int(image_res['preds'][i]),
            'image_prob': float(image_res['probs'][i]),
            'multi_pred': int(multimodal_res['preds'][i]),
            'multi_prob': float(multimodal_res['probs'][i]),
        }

    print('\n=== Text model failed (examples) ===')
    obs = []
    for i in text_examples:
        d = describe_case(i)
        print('\n---')
        print(f"Index: {d['index']}, File: {d['image']}, True: {d['true']}")
        print(f"Text pred: {d['text_pred']} (p={d['text_prob']:.3f}), Image pred: {d['image_pred']} (p={d['image_prob']:.3f}), Multi pred: {d['multi_pred']} (p={d['multi_prob']:.3f})")
        print('Text snippet:', d['text'])
        # Observations
        if d['text_pred'] != d['true'] and d['image_pred'] == d['true'] and d['multi_pred'] == d['true']:
            note = 'Text fooled by wording; image helped and multimodal corrected.'
        elif d['text_pred'] != d['true'] and d['image_pred'] != d['true'] and d['multi_pred'] == d['true']:
            note = 'Both single modalities failed, fusion recovered.'
        elif d['text_pred'] != d['true'] and d['image_pred'] == d['true'] and d['multi_pred'] != d['true']:
            note = 'Image correct but fusion failed to use it.'
        else:
            note = 'Other case (needs manual inspection).'
        print('Observation:', note)
        obs.append(note)

    print('\n=== Image model failed (examples) ===')
    for i in image_examples:
        d = describe_case(i)
        print('\n---')
        print(f"Index: {d['index']}, File: {d['image']}, True: {d['true']}")
        print(f"Text pred: {d['text_pred']} (p={d['text_prob']:.3f}), Image pred: {d['image_pred']} (p={d['image_prob']:.3f}), Multi pred: {d['multi_pred']} (p={d['multi_prob']:.3f})")
        print('Text snippet:', d['text'])
        if d['image_pred'] != d['true'] and d['text_pred'] == d['true'] and d['multi_pred'] == d['true']:
            note = 'Image fooled; text helped and multimodal corrected.'
        elif d['image_pred'] != d['true'] and d['text_pred'] != d['true'] and d['multi_pred'] == d['true']:
            note = 'Both failed, fusion recovered.'
        elif d['image_pred'] != d['true'] and d['text_pred'] == d['true'] and d['multi_pred'] != d['true']:
            note = 'Text correct but fusion did not preserve it.'
        else:
            note = 'Other case (needs manual inspection).'
        print('Observation:', note)

    print('\nDone.')


if __name__ == '__main__':
    main()
