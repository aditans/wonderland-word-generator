"""
dataset.py - Dataset ingestion, cleaning, tokenization, and sequence construction
for the Wonderland Word Generator (Alice's Adventures in Wonderland).
"""

import os
import re
import json
import urllib.request
from collections import Counter
import numpy as np

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "https://www.gutenberg.org/files/11/11-0.txt"
]

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
RAW_FILE_PATH = os.path.join(DATA_RAW_DIR, "alice_in_wonderland.txt")
CLEAN_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "alice_clean.txt")
VOCAB_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "vocab.json")
METADATA_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, "dataset_metadata.json")


def download_dataset():
    """Downloads Alice's Adventures in Wonderland from Project Gutenberg."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE_PATH) and os.path.getsize(RAW_FILE_PATH) > 10000:
        print(f"[Dataset] Found cached raw text at {RAW_FILE_PATH}")
        with open(RAW_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    print("[Dataset] Downloading text from Project Gutenberg...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    last_err = None
    for url in GUTENBERG_URLS:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="ignore")
                with open(RAW_FILE_PATH, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[Dataset] Downloaded and cached {len(content)} characters to {RAW_FILE_PATH}")
                return content
        except Exception as e:
            last_err = e
            print(f"[Dataset] Attempt failed for {url}: {e}")

    raise RuntimeError(f"Could not download dataset from Gutenberg: {last_err}")


def strip_gutenberg_boilerplate(raw_text):
    """Removes Project Gutenberg headers and footers to keep only narrative text."""
    start_match = re.search(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw_text, re.IGNORECASE)
    end_match = re.search(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw_text, re.IGNORECASE)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)

    narrative = raw_text[start_idx:end_idx].strip()
    return narrative


def preprocess_text(text):
    """
    Cleans text: converts to lowercase, normalizes punctuation and whitespace,
    and returns tokenized words.
    """
    # Replace unicode quotes and hyphens/dashes
    text = text.replace("“", ' " ').replace("”", ' " ')
    text = text.replace("‘", " ' ").replace("’", " ' ")
    text = text.replace("—", " ")
    text = text.replace("--", " ")

    # Lowercase
    text = text.lower()

    # Extract word tokens with regex (alphanumeric words and common contractions)
    tokens = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)
    return tokens


def build_vocabulary(tokens, max_vocab_size=3000, min_freq=2):
    """
    Builds word-to-index and index-to-word mappings.
    Reserves index 0 for <PAD> and index 1 for <UNK>.
    """
    counts = Counter(tokens)
    # Filter by minimum frequency and limit to max_vocab_size
    frequent_words = [w for w, c in counts.most_common() if c >= min_freq]
    frequent_words = frequent_words[: max_vocab_size - 2]

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(frequent_words, start=2):
        word2idx[word] = idx

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, counts


def construct_sequences(tokens, word2idx, seq_length=5):
    """
    Constructs autoregressive input (X) and target (y) sequences.
    For token sequence [w0, w1, w2, w3, w4, w5],
    Input X = [w0, w1, w2, w3, w4], Target y = w5.
    """
    # Map tokens to indices with <UNK> fallback
    unk_idx = word2idx["<UNK>"]
    indices = [word2idx.get(tok, unk_idx) for tok in tokens]

    X, y = [], []
    for i in range(len(indices) - seq_length):
        X.append(indices[i: i + seq_length])
        y.append(indices[i + seq_length])

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def prepare_dataset(seq_length=5, max_vocab_size=3000, min_freq=2):
    """Full dataset preparation pipeline returning X, y, word2idx, idx2word, metadata."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    raw_text = download_dataset()
    narrative_text = strip_gutenberg_boilerplate(raw_text)

    # Save cleaned narrative
    with open(CLEAN_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(narrative_text)

    tokens = preprocess_text(narrative_text)
    word2idx, idx2word, counts = build_vocabulary(tokens, max_vocab_size=max_vocab_size, min_freq=min_freq)
    X, y = construct_sequences(tokens, word2idx, seq_length=seq_length)

    metadata = {
        "dataset_name": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "source_url": "https://www.gutenberg.org/ebooks/11",
        "raw_character_count": len(raw_text),
        "narrative_character_count": len(narrative_text),
        "total_tokens": len(tokens),
        "unique_raw_tokens": len(counts),
        "vocab_size": len(word2idx),
        "seq_length": seq_length,
        "total_sequences": len(X),
        "top_10_words": counts.most_common(10)
    }

    # Save metadata & vocab
    with open(VOCAB_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "idx2word": {str(k): v for k, v in idx2word.items()}}, f, indent=2)

    with open(METADATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Dataset] Processed {metadata['total_tokens']} tokens, built vocab of {metadata['vocab_size']} words.")
    print(f"[Dataset] Created {metadata['total_sequences']} sequences of length {seq_length}.")

    return X, y, word2idx, idx2word, metadata


if __name__ == "__main__":
    X, y, w2i, i2w, meta = prepare_dataset()
    print("Sample input:", [i2w[idx] for idx in X[0]], "-> target:", i2w[y[0]])
