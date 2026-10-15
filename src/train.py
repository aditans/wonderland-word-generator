"""
train.py - Controlled training pipeline for Simple RNN, Bidirectional RNN, LSTM, and GRU.
Tracks training time, loss, accuracy, and perplexity across epochs.
Saves model checkpoints, metrics, and training histories.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset import prepare_dataset
from models import get_all_models

# Ensure reproducible splits
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(model_name, model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128, lr=0.003):
    """Compiles and trains a single model while recording runtime and metric trajectories."""
    print(f"\n==========================================")
    print(f"  Training: {model_name}")
    print(f"==========================================")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    total_time = time.time() - start_time
    avg_epoch_time = total_time / epochs

    # Evaluate final validation performance
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

    # Perplexity = exp(cross_entropy_loss)
    train_ppl = float(np.exp(train_loss))
    val_ppl = float(np.exp(val_loss))

    # Save model weights/architecture
    save_path = os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_').lower()}.keras")
    model.save(save_path)
    print(f"[{model_name}] Saved model to {save_path}")

    # Format history metrics
    hist_dict = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "train_acc": [float(x) for x in history.history["accuracy"]],
        "val_acc": [float(x) for x in history.history["val_accuracy"]],
        "train_ppl": [float(np.exp(x)) for x in history.history["loss"]],
        "val_ppl": [float(np.exp(x)) for x in history.history["val_loss"]],
    }

    summary = {
        "model_name": model_name,
        "parameters": int(model.count_params()),
        "total_training_time_sec": round(total_time, 2),
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "final_train_loss": round(float(train_loss), 4),
        "final_val_loss": round(float(val_loss), 4),
        "final_train_acc": round(float(train_acc * 100), 2),
        "final_val_acc": round(float(val_acc * 100), 2),
        "final_train_ppl": round(train_ppl, 2),
        "final_val_ppl": round(val_ppl, 2),
    }

    return model, hist_dict, summary


def run_training_pipeline(epochs=15, batch_size=128, lr=0.003):
    """Executes identical training for all 4 architectures on Alice in Wonderland."""
    X, y, word2idx, idx2word, metadata = prepare_dataset(seq_length=5, max_vocab_size=3000, min_freq=2)
    vocab_size = len(word2idx)

    # 85% train, 15% validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"\n[Pipeline] Vocabulary Size: {vocab_size}")
    print(f"[Pipeline] Training sequences: {len(X_train):,}, Validation sequences: {len(X_val):,}")
    print(f"[Pipeline] Experimental conditions: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, EmbeddingDim=128, Units=128")

    all_models = get_all_models(vocab_size=vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5)

    all_histories = {}
    all_summaries = []
    trained_models = {}

    for name, model in all_models.items():
        trained_m, hist, summary = train_model(
            name, model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        trained_models[name] = trained_m
        all_histories[name] = hist
        all_summaries.append(summary)

    # Save comprehensive results
    history_file = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)

    summary_file = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    df_summary = pd.DataFrame(all_summaries)
    csv_file = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df_summary.to_csv(csv_file, index=False)

    print("\n=======================================================")
    print("               FINAL QUANTITATIVE METRICS              ")
    print("=======================================================")
    print(df_summary.to_string(index=False))
    print("=======================================================")

    return trained_models, all_histories, all_summaries, word2idx, idx2word


if __name__ == "__main__":
    run_training_pipeline(epochs=12, batch_size=128, lr=0.003)
