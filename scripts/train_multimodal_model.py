#!/usr/bin/env python3
"""
Train a late-fusion multimodal model that combines a text LSTM branch and an image CNN branch.

Procedure (keeps things simple and explicit):
- Build datasets for train and val using JSONL files (separate, do NOT mix)
- Text branch: Embedding -> LSTM (frozen)
- Image branch: small CNN (convolutional backbone frozen)
- Late fusion: concatenate branch outputs, then Dense layers (trainable)
- Freeze base layers (embedding+LSTM, conv layers) so only fusion layers train
- Train for N epochs (default 5) and save best model by validation accuracy

Usage:
    python scripts/train_multimodal_model.py \
        --train data/fakeddit_subset/training_data_fakeddit.jsonl \
        --val data/fakeddit_subset/validation_data_fakeddit.jsonl \
        --img-dir data/fakeddit_subset/image_folder \
        --val-img-dir data/fakeddit_subset/validation_image \
        --tokenizer outputs/tokenizer.pkl --pad 128 --out models --epochs 5
"""

from pathlib import Path
import argparse
import json
import re
import pickle
from typing import List, Tuple, Any
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

# SimpleTokenizer compatibility so we can unpickle tokenizer.pkl
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


# --- Data helpers (same rules as previous scripts) ---

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
            # optionally skip entire dict if it's a model role (answers)
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
    # Prefer explicit model answers when present
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


def preprocess_image(filename, label, image_size=(128, 128)):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label


def build_dataset_from_jsonl(jsonl_path: Path, img_dir: Path, tokenizer, pad: int, exclude_model_answers: bool = False) -> Tuple[np.ndarray, List[Path], np.ndarray, List[str]]:
    records = load_jsonl(jsonl_path)
    texts = []
    img_paths = []
    labels = []
    raw_texts = []
    for rec in records:
        txt = extract_text(rec, exclude_model_answers=exclude_model_answers)
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
    X_text = pad_sequences(seqs, maxlen=pad, padding='post')
    y = np.array(labels, dtype=np.float32)
    return X_text, img_paths, y, raw_texts


# --- Model builders ---

def build_text_branch(vocab_size: int, embed_dim: int, lstm_units: int, input_length: int):
    text_input = layers.Input(shape=(input_length,), dtype='int32', name='text_input')
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name='text_embedding')(text_input)
    x = layers.LSTM(lstm_units, name='text_lstm')(x)
    return text_input, x


def build_image_branch(input_shape=(128, 128, 3), name_prefix='img'):
    img_input = layers.Input(shape=input_shape, name='image_input')
    y = layers.Conv2D(32, (3, 3), activation='relu', name=f'{name_prefix}_conv1')(img_input)
    y = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(y)
    y = layers.Conv2D(64, (3, 3), activation='relu', name=f'{name_prefix}_conv2')(y)
    y = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(y)
    y = layers.Conv2D(128, (3, 3), activation='relu', name=f'{name_prefix}_conv3')(y)
    y = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool3')(y)
    y = layers.Flatten(name=f'{name_prefix}_flatten')(y)
    y = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense')(y)
    return img_input, y


def main():
    parser = argparse.ArgumentParser(description='Train a multimodal (text+image) late-fusion model')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--img-dir', type=Path, default=Path('data/fakeddit_subset/image_folder'))
    parser.add_argument('--val-img-dir', type=Path, default=Path('data/fakeddit_subset/validation_image'))
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/tokenizer.pkl'))
    parser.add_argument('--pad', type=int, default=128)
    parser.add_argument('--out', type=Path, default=Path('models'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--lstm', type=int, default=128)
    parser.add_argument('--image-size', type=int, nargs=2, default=(128, 128))
    parser.add_argument('--image-compress-dim', type=int, default=64, help='Dimension to compress image features before fusion')
    parser.add_argument('--text-expand-dim', type=int, default=256, help='Dimension to expand text features before fusion')
    parser.add_argument('--fusion-dim1', type=int, default=256, help='Fusion layer size 1')
    parser.add_argument('--fusion-dim2', type=int, default=64, help='Fusion layer size 2')
    parser.add_argument('--exclude-model-answers', action='store_true', help='Exclude model role answers from extracted text to avoid label leakage')
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise FileNotFoundError(f'Tokenizer not found: {args.tokenizer}')
    with args.tokenizer.open('rb') as f:
        tokenizer = pickle.load(f)

    print('Building datasets...')
    X_train_text, train_img_paths, y_train, raw_train_texts = build_dataset_from_jsonl(args.train, args.img_dir, tokenizer, pad=args.pad, exclude_model_answers=args.exclude_model_answers)
    X_val_text, val_img_paths, y_val, raw_val_texts = build_dataset_from_jsonl(args.val, args.val_img_dir, tokenizer, pad=args.pad, exclude_model_answers=args.exclude_model_answers)

    print(f'Train items: {len(y_train)}, Val items: {len(y_val)}')
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError('No labeled multimodal samples found. Check JSONL and image folders.')

    # Create tf.data datasets that yield (text_seq, image_tensor), label
    def make_tf_dataset(X_text, img_paths, labels, batch_size, image_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((X_text, img_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(labels))

        def _map(text_seq, img_path, label):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            img = img / 255.0
            return ({'text_input': text_seq, 'image_input': img}, label)

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_tf_dataset(X_train_text, train_img_paths, y_train, batch_size=args.batch, image_size=tuple(args.image_size), shuffle=True)
    val_ds = make_tf_dataset(X_val_text, val_img_paths, y_val, batch_size=args.batch, image_size=tuple(args.image_size), shuffle=False)

    # Build branches
    text_input, text_feat = build_text_branch(vocab_size=len(tokenizer.word_index), embed_dim=args.embed, lstm_units=args.lstm, input_length=args.pad)
    img_input, img_feat = build_image_branch(input_shape=(args.image_size[0], args.image_size[1], 3))

    # Freeze base branches
    # Freeze embedding & LSTM by making their layers non-trainable
    # (they are named text_embedding and text_lstm)

    # Build a model for the branches to make it easy to control layer.trainable
    text_branch = models.Model(text_input, text_feat, name='text_branch')
    image_branch = models.Model(img_input, img_feat, name='image_branch')

    # Freeze layers in branches
    for layer in text_branch.layers:
        layer.trainable = False
    for layer in image_branch.layers:
        # freeze convolutional backbone and dense projection
        layer.trainable = False

    # Apply small trainable transforms to branch outputs (compress image, expand text)
    img_trans = layers.Dense(args.image_compress_dim, activation='relu', name='image_compress')(image_branch.output)
    txt_trans = layers.Dense(args.text_expand_dim, activation='relu', name='text_expand')(text_branch.output)

    # Concatenate the transformed features and apply fusion layers (only fusion layers are trainable)
    combined = layers.Concatenate(name='fusion_concat')([txt_trans, img_trans])
    z = layers.Dense(args.fusion_dim1, activation='relu', name='fusion_dense1')(combined)
    z = layers.Dropout(0.5, name='fusion_dropout')(z)
    z = layers.Dense(args.fusion_dim2, activation='relu', name='fusion_dense2')(z)
    out = layers.Dense(1, activation='sigmoid', name='out')(z)

    model = models.Model(inputs=[text_branch.input, image_branch.input], outputs=out, name='multimodal_model')

    # Compile only trainable layers (the base branches are frozen)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / 'multimodal_model.keras'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(best_path),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

    print('Training multimodal fusion layers (base branches are frozen)...')
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[checkpoint])

    print('\nTraining history:')
    for epoch in range(len(history.history['loss'])):
        t_loss = history.history['loss'][epoch]
        v_loss = history.history['val_loss'][epoch]
        t_acc = history.history['accuracy'][epoch]
        v_acc = history.history['val_accuracy'][epoch]
        print(f'Epoch {epoch+1}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, train_acc={t_acc:.4f}, val_acc={v_acc:.4f}')

    # Evaluate final model on validation
    print('\nEvaluating model on validation set...')
    val_probs = model.predict(val_ds).ravel()
    # re-construct flat true labels in dataset order
    val_true = np.array(y_val)
    val_preds = (val_probs >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    acc = accuracy_score(val_true, val_preds)
    cm = confusion_matrix(val_true, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary')
    print('\n=== Validation Summary ===')
    print(f'Accuracy: {acc:.4f}')
    print('Confusion matrix:\n', cm)
    print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')

    # Save model (Keras and SavedModel)
   
    keras_path = args.out / 'multimodal_model.keras'
    model.save(keras_path)
    print(f'Model saved to {keras_path}')


    print('Done.')


if __name__ == '__main__':
    main()
