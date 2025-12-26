#!/usr/bin/env python3
"""
Load `training_data_fakeddit.jsonl` and print a few samples for quick text EDA.

Usage:
    python text_eda.py          # prints 5 samples from default path
    python text_eda.py -p path/to/file.jsonl -n 10
"""

import json
from pathlib import Path
import argparse
import re
import string
from collections import Counter

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "fakeddit_subset" / "training_data_fakeddit.jsonl"


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse JSON on line {i}: {e}")
    return data


def print_samples(data, n=5, max_field_len=300):
    print(f"Total records loaded: {len(data)}")
    for idx, rec in enumerate(data[:n], start=1):
        print(f"\n--- Sample {idx} ---")
        if not isinstance(rec, dict):
            print(repr(rec))
            continue
        for k, v in rec.items():
            if isinstance(v, str):
                text = v.replace("\n", " ")
                if len(text) > max_field_len:
                    text = text[:max_field_len] + "...[truncated]"
                print(f"{k}: {text}")
            else:
                print(f"{k}: {repr(v)}")


def extract_all_text(obj):
    """Recursively extract all string values from a nested record and join them."""
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

    walk(obj)
    return " ".join(t.replace("\n", " ") for t in texts if t)


def analyze_text(text, top_n=10):
    # basic cleaning and tokenization
    tokens = re.findall(r"\b\w+\b", text)
    words = [t for t in tokens]

    char_len = len(text)
    word_count = len(words)
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0

    # caps / ALL-CAPS tokens
    allcaps_tokens = [w for w in words if w.isalpha() and w.isupper() and len(w) > 1]
    allcaps_count = len(allcaps_tokens)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / sum(1 for c in text if c.isalpha()) if any(c.isalpha() for c in text) else 0

    # punctuation & noise
    quote_count = text.count('"') + text.count("'")
    question_marks = text.count('?')
    exclamation_marks = text.count('!')
    ellipses = text.count('...')
    urls = len(re.findall(r"https?://\S+|www\.\S+", text))
    hashtags = len(re.findall(r"#\w+", text))
    mentions = len(re.findall(r"@\w+", text))
    punctuation_count = sum(1 for c in text if c in string.punctuation)

    # emoji detection (basic unicode ranges)
    emoji_pattern = re.compile("[\U0001F300-\U0001F6FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]", flags=re.UNICODE)
    emojis = emoji_pattern.findall(text)
    emoji_count = len(emojis)

    # fake/real indicative words
    fake_indicators = [
        'fake', 'hoax', 'photoshop', 'photoshopped', 'edited', 'satire', 'clickbait', 'not real', 'false', 'rumor'
    ]
    real_indicators = ['real', 'true', 'genuine', 'authentic']
    fake_hits = {w: text.lower().count(w) for w in fake_indicators if w in text.lower()}
    real_hits = {w: text.lower().count(w) for w in real_indicators if w in text.lower()}

    # top words
    cnt = Counter(w.lower() for w in words if w)
    top_words = cnt.most_common(top_n)

    return {
        'char_len': char_len,
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'allcaps_count': allcaps_count,
        'uppercase_ratio': round(uppercase_ratio, 3),
        'quote_count': quote_count,
        'question_marks': question_marks,
        'exclamation_marks': exclamation_marks,
        'ellipses': ellipses,
        'urls': urls,
        'hashtags': hashtags,
        'mentions': mentions,
        'punctuation_count': punctuation_count,
        'emoji_count': emoji_count,
        'fake_hits': fake_hits,
        'real_hits': real_hits,
        'top_words': top_words,
    }


def print_analysis(analysis):
    print("\n--- Text Analysis ---")
    print(f"Characters: {analysis['char_len']}")
    print(f"Words: {analysis['word_count']}, Avg word len: {analysis['avg_word_len']:.2f}")
    print(f"ALL-CAPS tokens: {analysis['allcaps_count']}, Uppercase ratio: {analysis['uppercase_ratio']}")
    print(f"Quotes: {analysis['quote_count']}, Questions: {analysis['question_marks']}, Exclaims: {analysis['exclamation_marks']}, Ellipses: {analysis['ellipses']}")
    print(f"URLs: {analysis['urls']}, Hashtags: {analysis['hashtags']}, Mentions: {analysis['mentions']}, Emojis: {analysis['emoji_count']}")
    print(f"Punctuation count: {analysis['punctuation_count']}")
    if analysis['fake_hits']:
        print(f"Fake-indicative words: {analysis['fake_hits']}")
    if analysis['real_hits']:
        print(f"Real-indicative words: {analysis['real_hits']}")
    print(f"Top words: {analysis['top_words']}")


def analyze_first(data):
    rec = data[0]
    text = extract_all_text(rec)
    print("\n=== Sample 1 (joined text fields) ===")
    print(text[:1000] + ("...[truncated]" if len(text) > 1000 else ""))
    analysis = analyze_text(text)
    print_analysis(analysis)


def main():
    parser = argparse.ArgumentParser(description="Load JSONL and print sample records for quick EDA")
    parser.add_argument("-p", "--path", type=Path, default=DEFAULT_PATH, help="Path to JSONL file")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of samples to print")
    parser.add_argument("--analyze-first", action="store_true", help="Analyze only the first sample and print features")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: file not found: {args.path}")
        return

    data = load_jsonl(args.path)
    if not data:
        print("No records loaded from file.")
        return

    if args.analyze_first:
        analyze_first(data)
    else:
        print_samples(data, n=args.num)


if __name__ == "__main__":
    main()
