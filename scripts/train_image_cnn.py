#!/usr/bin/env python3
"""
Train a simple CNN classifier on images from the Fakeddit subset.

What it does (simple, readable):
- Loads train and validation JSONL files
- Extracts image filenames and labels (0/1)
- Builds tf.data datasets that load images from disk and preprocess (resize, normalize)
- Defines a small CNN: Conv -> Pool -> Conv -> Pool -> Dense -> Dropout -> Sigmoid
- Trains while printing training/validation loss and accuracy
- Saves the best model by validation accuracy to `models/cnn_model.keras` and exports SavedModel

Usage:
    python scripts/train_image_cnn.py --train data/fakeddit_subset/training_data_fakeddit.jsonl \
        --val data/fakeddit_subset/validation_data_fakeddit.jsonl --img-dir data/fakeddit_subset/image_folder --out models --epochs 5

Note: This script does not do advanced augmentation or hyperparameter search; it's intentionally simple to observe behavior.
"""

from pathlib import Path
import argparse
import json
import re
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Any


def load_jsonl(path: Path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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
    joined = ' '.join(label_candidates).lower()
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


def build_file_label_list(jsonl_path: Path, img_dir: Path) -> Tuple[List[Path], List[int]]:
    records = load_jsonl(jsonl_path)
    files = []
    labels = []
    for rec in records:
        img_id = extract_image_id(rec)
        if not img_id:
            continue
        img_path = img_dir / img_id
        if not img_path.exists():
            # skip if image not present
            continue
        lab = extract_label(rec)
        if lab not in (0, 1):
            continue
        files.append(img_path)
        labels.append(lab)
    return files, labels


def preprocess_image(filename, label, image_size=(128, 128)):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label


def make_dataset(filepaths: List[Path], labels: List[int], batch_size: int, image_size=(128, 128), shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in filepaths], labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.map(lambda f, l: preprocess_image(f, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape=(128, 128, 3)):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train a simple CNN on images')
    parser.add_argument('--train', type=Path, required=True, help='Training JSONL')
    parser.add_argument('--val', type=Path, required=True, help='Validation JSONL')
    parser.add_argument('--img-dir', type=Path, default=Path('data/fakeddit_subset/image_folder'), help='Image folder for train images')
    parser.add_argument('--val-img-dir', type=Path, default=Path('data/fakeddit_subset/validation_image'), help='Image folder for validation images')
    parser.add_argument('--out', type=Path, default=Path('models'), help='Model output dir')
    parser.add_argument('--image-size', type=int, nargs=2, default=(128, 128), help='Image size (width height)')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # prepare lists
    print('Building file lists...')
    train_files, train_labels = build_file_label_list(args.train, args.img_dir)
    val_files, val_labels = build_file_label_list(args.val, args.val_img_dir)
    print(f'Found {len(train_files)} training images, {len(val_files)} validation images')
    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError('No images found for train/val. Check paths and JSONL extraction rules.')

    train_ds = make_dataset(train_files, train_labels, batch_size=args.batch, image_size=tuple(args.image_size), shuffle=True)
    val_ds = make_dataset(val_files, val_labels, batch_size=args.batch, image_size=tuple(args.image_size), shuffle=False)

    # model
    model = build_model(input_shape=(args.image_size[0], args.image_size[1], 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # callbacks: save best by val_accuracy
    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / 'cnn_model.keras'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor='val_accuracy', save_best_only=True, verbose=1)

    print('Training...')
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[checkpoint])

    # print observed metrics per epoch
    print('\nTraining history:')
    for epoch in range(len(history.history['loss'])):
        t_loss = history.history['loss'][epoch]
        v_loss = history.history['val_loss'][epoch]
        t_acc = history.history['accuracy'][epoch]
        v_acc = history.history['val_accuracy'][epoch]
        print(f'Epoch {epoch+1}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, train_acc={t_acc:.4f}, val_acc={v_acc:.4f}')

    # Evaluate on validation set and report a few mistakes
    print('\nEvaluating on validation set...')
    val_probs = model.predict(val_ds).ravel()
    # build flat arrays of true labels in order of val_ds
    val_true = np.array([lab for lab in val_labels])
    # Need to get preds aligned with dataset order (shuffle=False ensures order)
    val_preds = (val_probs >= 0.5).astype(int)

    # Basic metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    acc = accuracy_score(val_true, val_preds)
    cm = confusion_matrix(val_true, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary')
    print('\n=== Validation Summary ===')
    print(f'Accuracy: {acc:.4f}')
    print('Confusion matrix:\n', cm)
    print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')

    # show up to 8 wrong images
    wrong_idx = np.where(val_preds != val_true)[0]
    print(f'\nWrong predictions: {len(wrong_idx)}')
    for i in wrong_idx[:8]:
        print('---')
        print(f'File: {val_files[i].name}, True: {val_true[i]}, Pred: {val_preds[i]}, Prob: {val_probs[i]:.4f}')

    # Save exported SavedModel dir as well
    saved_dir = args.out / 'cnn_model'
    try:
        model.export(saved_dir)
        print(f'Model exported to {saved_dir}')
    except Exception:
        tf.saved_model.save(model, str(saved_dir))
        print(f'Model saved via tf.saved_model.save to {saved_dir}')

    print('Done.')


if __name__ == '__main__':
    main()
