#!/usr/bin/env python3
"""
Preprocess FakeNews (Fakeddit subset) JSONL into tokenized text sequences,
labels and image ids.

Outputs (by default, saved into `outputs/`):
- text_sequnece.npy           : numpy array of token id sequences (dtype=object) or padded 2D array when `--pad` is used
- labels.npy                  : numpy array of integer labels (1=fake, 0=real, -1=unknown)
- image_ids.csv               : CSV (image_id,label)
- outputs/tokenizer.pkl       : pickled tokenizer object

Usage:
    python scripts/text_preprocessing.py --path data/fakeddit_subset/training_data_fakeddit.jsonl --out outputs --min-count 2 --pad 128

The tokenizer is a small, clear class (SimpleTokenizer) that builds a vocabulary from the
training texts and converts texts to sequences of integer ids.
"""

from pathlib import Path
import json
import re
import argparse
import pickle
from collections import Counter
import numpy as np
from typing import List, Tuple, Dict, Any

# --- Helpers to read and extract ---

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


def clean_text(text: str) -> str:
    """Clean prompt-like tokens from text.
    - Remove question prompts like 'is the picture ... ?' and 'Only Answer "Yes"/"No"'
    - Extract title contents when present (Title:"...") and keep that text
    - Remove short single-answer tokens like 'Yes'/'No' coming from model responses
    - Normalize whitespace
    """
    if not text:
        return text
    # extract quoted Title if present: keep the quoted content
    m = re.search(r'Title:\s*"([^"]+)"', text, flags=re.I)
    if m:
        text = m.group(1)
    # remove parenthetical Only Answer patterns
    text = re.sub(r"\(Only Answer\s*\"?Yes\"?\/?\"?No\"?\)", "", text, flags=re.I)
    # remove explicit 'Only Answer "Yes"/"No"' without parentheses
    text = re.sub(r"Only Answer\s*\"?Yes\"?\/?\"?No\"?", "", text, flags=re.I)
    # remove the QA question like 'is the picture and title indicate that this is from a fake news?'
    text = re.sub(r"\bis the picture[^\n\r]*\?", "", text, flags=re.I)
    # remove trailing 'model Yes' or bare 'Yes'/'No' tokens that are likely answers
    text = re.sub(r"\bmodel\b\s*\w*", "", text, flags=re.I)
    text = re.sub(r"\b(yes|no)\b", "", text, flags=re.I)
    # normalize spaces and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text(record: Any, exclude_model_answers: bool = False) -> str:
    """Extract textual content from a record while avoiding fileUri/image paths, and clean it.

    If exclude_model_answers is True, skip any dict that has role == 'model' to avoid
    including model-generated answers (these may contain the label and cause leakage).
    """
    texts = []

    def walk(o):
        if isinstance(o, str):
            # skip raw URIs and obvious file paths (avoid embedding file uris into text)
            if '://' in o or 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)", o, re.I):
                return
            texts.append(o.replace("\n", " "))
        elif isinstance(o, dict):
            # optionally skip any 'model' role dict entirely
            if exclude_model_answers:
                role = (o.get('role') or '').lower() if isinstance(o.get('role'), str) else ''
                if role == 'model':
                    return

            for k, v in o.items():
                # prefer explicit text fields
                if k.lower() in ('text', 'title', 'caption') and isinstance(v, str):
                    texts.append(v.replace("\n", " "))
                else:
                    walk(v)
        elif isinstance(o, list):
            for e in o:
                walk(e)

    walk(record)
    # join, clean and normalize spaces
    joined = " ".join(t for t in texts if t).strip()
    return clean_text(joined)


def extract_image_id(record: Any) -> str:
    """Find the first image filename/URI in the record and return a basename string or empty str."""
    found = None

    def walk(o):
        nonlocal found
        if found:
            return
        if isinstance(o, str):
            if 'gs://' in o or re.search(r"\.(jpg|jpeg|png|gif)($|\?)", o, re.I) or o.startswith('http'):
                # take basename
                found = Path(o.split('://')[-1].split('/')[-1].split('?')[0]).name
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v)
                if found:
                    return
        elif isinstance(o, list):
            for e in o:
                walk(e)
                if found:
                    return

    walk(record)
    return found or ""


def extract_label(record: Any) -> int:
    """Extract label from the record.
    Prefer an explicit "model" role answer when present; otherwise fallback to keyword matching.
    Returns: 1 (fake), 0 (real), or -1 if unknown.
    """
    # 1) Look for explicit model responses (preferred)
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

    # 2) Fallback: gather text and search keywords (avoid using the prompt 'Only Answer "Yes"/"No"' as evidence)
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

    # look for explicit fake indicators first
    if re.search(r"\bfake\b|\bhoax\b|\bphotoshop\b|\bphotoshopped\b", joined):
        return 1
    if re.search(r"\breal\b|\btrue\b|\bgenuine\b|\bauthentic\b", joined):
        return 0

    # numeric label fallback
    m2 = re.search(r"\blabel[:=]\s*(\d+)\b", joined)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            pass

    return -1


# --- Simple tokenizer ---
class SimpleTokenizer:
    def __init__(self, lower=True, token_pattern=r"\b\w+\b"):
        self.lower = lower
        self.token_pattern = re.compile(token_pattern)
        self.word_counts = Counter()
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_word = {0: "<PAD>", 1: "<UNK>"}
        self.fitted = False

    def tokenize(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        return self.token_pattern.findall(text)

    def fit_on_texts(self, texts: List[str], min_count: int = 1, max_vocab: int = None):
        for t in texts:
            self.word_counts.update(self.tokenize(t))

        # filter and sort
        items = [(w, c) for w, c in self.word_counts.items() if c >= min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab:
            items = items[:max_vocab]

        # build indices starting from 2 (0 PAD, 1 UNK)
        idx = 2
        for w, _ in items:
            if w not in self.word_index:
                self.word_index[w] = idx
                self.index_word[idx] = w
                idx += 1
        self.fitted = True

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit_on_texts first.")
        seqs = []
        for t in texts:
            seq = [self.word_index.get(w, self.word_index['<UNK>']) for w in self.tokenize(t)]
            seqs.append(seq)
        return seqs

    def save(self, path: Path):
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> 'SimpleTokenizer':
        with path.open('rb') as f:
            return pickle.load(f)


# --- Main pipeline ---

def build_dataset(path: Path, exclude_model_answers: bool = False) -> Tuple[List[str], List[int], List[str]]:
    records = load_jsonl(path)
    texts = []
    labels = []
    image_ids = []
    for rec in records:
        txt = extract_text(rec, exclude_model_answers=exclude_model_answers)
        img = extract_image_id(rec)
        lab = extract_label(rec)
        # Skip entries with neither text nor image to keep dataset useful
        if not txt and not img:
            continue
        texts.append(txt)
        labels.append(lab)
        image_ids.append(img)
    return texts, labels, image_ids


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


def save_outputs(outdir: Path, sequences, labels, image_ids, tokenizer: SimpleTokenizer, pad=None):
    outdir.mkdir(parents=True, exist_ok=True)
    # Save sequences as object dtype if variable length, otherwise as 2D padded array
    text_seq_path = outdir / 'text_sequnece.npy'  # note: uses user's requested filename
    if pad:
        padded = pad_sequences(sequences, maxlen=pad, padding='post')
        np.save(text_seq_path, padded)
        print(f"Saved padded sequences to {text_seq_path} (shape={padded.shape})")
    else:
        np.save(text_seq_path, np.array(sequences, dtype=object))
        print(f"Saved variable-length sequences to {text_seq_path} (dtype=object)")

    labels_path = outdir / 'labels.npy'
    np.save(labels_path, np.array(labels, dtype=np.int32))
    print(f"Saved labels to {labels_path}")

    csv_path = outdir / 'image_ids.csv'
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('image_id,label\n')
        for iid, lab in zip(image_ids, labels):
            f.write(f"{iid},{lab}\n")
    print(f"Saved image ids CSV to {csv_path}")

    # Save tokenizer (single canonical file)
    tk_path = outdir / 'tokenizer.pkl'
    tokenizer.save(tk_path)
    print(f"Saved tokenizer to {tk_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract text sequences, labels and image ids and build tokenizer")
    parser.add_argument('--path', type=Path, required=True, help='Path to training JSONL file')
    parser.add_argument('--out', type=Path, default=Path('outputs'), help='Output directory')
    parser.add_argument('--min-count', type=int, default=1, help='Minimum token count for vocab')
    parser.add_argument('--max-vocab', type=int, default=None, help='Maximum vocabulary size (optional)')
    parser.add_argument('--pad', type=int, default=None, help='Pad sequences to fixed length (optional)')
    parser.add_argument('--exclude-model-answers', action='store_true', help='Exclude model role answers from extracted text to avoid label leakage')
    args = parser.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}")
        return

    print(f"Building dataset from {args.path} ...")
    texts, labels, image_ids = build_dataset(args.path, exclude_model_answers=args.exclude_model_answers)
    print(f"Found {len(texts)} items (texts), {len(image_ids)} image ids")

    print("Fitting tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(texts, min_count=args.min_count, max_vocab=args.max_vocab)
    print(f"Vocab size (including PAD/UNK): {len(tokenizer.word_index)}")

    sequences = tokenizer.texts_to_sequences(texts)

    save_outputs(args.out, sequences, labels, image_ids, tokenizer, pad=args.pad)
    print("Done.")


if __name__ == '__main__':
    main()
