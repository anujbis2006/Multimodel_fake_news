#!/usr/bin/env python3
"""
Streamlit app for Multimodal Fake News Detection
- Loads tokenizer and multimodal model (.keras only)
- Lets user input text and upload an image
- Uses same preprocessing as training
"""

from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import re
import io
import json
import random

# ---------------------------
# Config (MUST match training)
# ---------------------------
TOKENIZER_PATH = Path("outputs/tokenizer.pkl")
MODEL_PATH = Path("models/multimodal_model.keras")

MAX_SEQ_LEN = 128
IMAGE_SIZE = (128, 128)

VAL_JSON = Path("data/fakeddit_subset/validation_data_fakeddit.jsonl")
VAL_IMG_DIR = Path("data/fakeddit_subset/validation_image")

# ---------------------------
# Utilities
# ---------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'Title:\s*"([^"]+)"', text, flags=re.I)
    if m:
        text = m.group(1)
    text = re.sub(r"\(Only Answer.*?\)", "", text, flags=re.I)
    text = re.sub(r"\bis the picture[^\n\r]*\?", "", text, flags=re.I)
    text = re.sub(r"\b(yes|no)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pad_sequence(seq, maxlen=MAX_SEQ_LEN):
    arr = np.zeros((maxlen,), dtype=np.int32)
    arr[: min(len(seq), maxlen)] = seq[:maxlen]
    return arr


# ---------------------------
# Cached loading
# ---------------------------

@st.cache_resource
def load_tokenizer(path: Path):
    class SimpleTokenizerCompat:
        def __init__(self, lower=True, token_pattern=r"\b\w+\b"):
            import re
            from collections import Counter
            self.lower = lower
            self.token_pattern = re.compile(token_pattern)
            self.word_index = {"<PAD>": 0, "<UNK>": 1}

        def tokenize(self, text):
            if self.lower:
                text = text.lower()
            return self.token_pattern.findall(text or "")

        def texts_to_sequences(self, texts):
            return [
                [self.word_index.get(w, 1) for w in self.tokenize(t)]
                for t in texts
            ]

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "SimpleTokenizer":
                return SimpleTokenizerCompat
            return super().find_class(module, name)

    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return CompatUnpickler(f).load()


@st.cache_resource
def load_multimodal_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_model(path)


# ---------------------------
# Preprocessing
# ---------------------------

def preprocess_text(text, tokenizer):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])[0]
    return pad_sequence(seq)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


# ---------------------------
# Validation helpers
# ---------------------------

@st.cache_data
def load_validation_records():
    records = []
    if not VAL_JSON.exists():
        return records
    with VAL_JSON.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def find_image_name(record):
    def walk(o):
        if isinstance(o, str) and re.search(r"\.(jpg|png|jpeg)", o, re.I):
            return Path(o).name
        if isinstance(o, dict):
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        if isinstance(o, list):
            for e in o:
                r = walk(e)
                if r:
                    return r
        return None

    return walk(record)


def extract_text_from_record(record):
    texts = []

    def walk(o):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for e in o:
                walk(e)

    walk(record)
    return " ".join(texts)


def load_random_example():
    recs = load_validation_records()
    random.shuffle(recs)
    for r in recs:
        img_name = find_image_name(r)
        if not img_name:
            continue
        img_path = VAL_IMG_DIR / img_name
        if not img_path.exists():
            continue
        text = extract_text_from_record(r)
        return text, img_path
    return None, None


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config("Multimodal Fake News Detection")
    st.title("📰 Multimodal Fake News Detection")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_multimodal_model(MODEL_PATH)

    st.success("Model and tokenizer loaded")

    if st.button("Load example from validation set"):
        text, img_path = load_random_example()
        if text:
            st.session_state["text"] = text
            st.session_state["img"] = img_path.read_bytes()

    text_input = st.text_area(
        "Enter news text",
        value=st.session_state.get("text", ""),
        height=150,
    )

    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if "img" in st.session_state and not uploaded:
        st.image(st.session_state["img"], caption="Example image", width=250)

    if st.button("Predict"):
        if not text_input.strip():
            st.error("Please enter text")
            return

        if not uploaded and "img" not in st.session_state:
            st.error("Please upload an image")
            return

        img = (
            Image.open(uploaded)
            if uploaded
            else Image.open(io.BytesIO(st.session_state["img"]))
        )

        t = preprocess_text(text_input, tokenizer)
        i = preprocess_image(img)

        t = np.expand_dims(t, 0)
        i = np.expand_dims(i, 0)

        prob = float(model.predict([t, i], verbose=0)[0][0])

        label = "FAKE NEWS" if prob >= 0.5 else "REAL NEWS"
        confidence = prob if prob >= 0.5 else 1 - prob

        if label == "FAKE NEWS":
            st.error(f"🚨 {label} — Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ {label} — Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
