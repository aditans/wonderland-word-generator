"""
generate.py - Autoregressive text generation with temperature-scaled sampling.
Performs qualitative evaluation across Simple RNN, Bidirectional RNN, LSTM, and GRU
using identical seed sequences from Alice's Adventures in Wonderland under 3 temperatures.
Calculates diversity, repetition rate, and qualitative assessment metrics.
"""

import os
import json
import numpy as np
import tensorflow as tf

from dataset import preprocess_text

SEED_PHRASES = [
    "alice was beginning to get",
    "the cat only grinned when",
    "down the rabbit hole went",
    "the hatter opened his eyes"
]

TEMPERATURES = [0.5, 0.8, 1.2]
GEN_LENGTH = 20  # Number of words to generate sequentially


def sample_with_temperature(logits, temperature=1.0):
    """
    Applies temperature scaling to logits and draws a categorical sample.
    T < 1.0: sharper, more conservative / deterministic predictions.
    T = 1.0: default model confidence distribution.
    T > 1.0: softer, higher entropy, more diverse / chaotic exploration.
    """
    logits = np.asarray(logits, dtype=np.float64)

    # Avoid zero or negative temperature
    temperature = max(float(temperature), 1e-4)

    if temperature < 0.1:
        return int(np.argmax(logits))

    # Apply temperature
    scaled_logits = logits / temperature
    # Numerical stability
    scaled_logits -= np.max(scaled_logits)
    exp_logits = np.exp(scaled_logits)
    probs = exp_logits / np.sum(exp_logits)

    # Sample from multinomial distribution
    choice = np.random.choice(len(probs), p=probs)
    return int(choice)


def generate_text_sequence(model, seed_text, word2idx, idx2word, seq_length=5, gen_words=20, temperature=1.0):
    """
    Sequentially generates `gen_words` given `seed_text` via autoregressive rolling window.
    """
    tokens = preprocess_text(seed_text)
    unk_idx = word2idx.get("<UNK>", 1)

    # Pad or truncate seed tokens to match exact seq_length
    if len(tokens) < seq_length:
        tokens = ["<PAD>"] * (seq_length - len(tokens)) + tokens
    else:
        tokens = tokens[-seq_length:]

    current_sequence = [word2idx.get(t, unk_idx) for t in tokens]
    generated_tokens = []

    for _ in range(gen_words):
        input_arr = np.array([current_sequence[-seq_length:]], dtype=np.int32)
        preds = model(input_arr, training=False).numpy()[0]

        # Prevent predicting <PAD> and suppress <UNK>
        preds[0] = 1e-12
        if len(preds) > 1:
            preds[1] = preds[1] * 0.1

        # Re-convert probabilities to pseudo-logits for temperature sampling
        eps = 1e-12
        logits = np.log(np.maximum(preds, eps))

        next_idx = sample_with_temperature(logits, temperature=temperature)
        next_word = idx2word.get(str(next_idx), idx2word.get(next_idx, "<UNK>"))

        generated_tokens.append(next_word)
        current_sequence.append(next_idx)

    full_generated_text = seed_text + " " + " ".join(generated_tokens)
    return full_generated_text, generated_tokens


def calculate_text_metrics(tokens):
    """Computes diversity (type-token ratio) and repetition rate (bi-gram duplication)."""
    if not tokens:
        return {"diversity": 0.0, "repetition_rate": 0.0}

    total_words = len(tokens)
    unique_words = len(set(tokens))
    diversity = round(unique_words / total_words, 3)

    # Bigram repetition count
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    if bigrams:
        dup_bigrams = len(bigrams) - len(set(bigrams))
        rep_rate = round(dup_bigrams / len(bigrams), 3)
    else:
        rep_rate = 0.0

    return {
        "diversity": diversity,
        "repetition_rate": rep_rate,
        "unique_words": unique_words,
        "total_words": total_words
    }


def evaluate_qualitative_generation(trained_models, word2idx, idx2word,
                                    seed_phrases=SEED_PHRASES,
                                    temperatures=TEMPERATURES,
                                    gen_words=20):
    """
    Runs text generation across all models, seeds, and temperatures.
    Produces comprehensive qualitative and quantitative analysis data.
    """
    results = {}
    table_rows = []

    for model_name, model in trained_models.items():
        results[model_name] = {}
        for temp in temperatures:
            results[model_name][str(temp)] = []
            for seed in seed_phrases:
                full_text, gen_tokens = generate_text_sequence(
                    model, seed, word2idx, idx2word,
                    seq_length=5, gen_words=gen_words, temperature=temp
                )
                metrics = calculate_text_metrics(gen_tokens)

                entry = {
                    "seed": seed,
                    "temperature": temp,
                    "generated_continuation": " ".join(gen_tokens),
                    "full_passage": full_text,
                    "diversity": metrics["diversity"],
                    "repetition_rate": metrics["repetition_rate"]
                }
                results[model_name][str(temp)].append(entry)

                table_rows.append({
                    "Model": model_name,
                    "Temp": temp,
                    "Seed": seed,
                    "Generated Passage": full_text,
                    "Diversity": metrics["diversity"],
                    "Repetition Rate": metrics["repetition_rate"]
                })

    # Save outputs
    out_json = os.path.join("results", "generation_samples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, table_rows
