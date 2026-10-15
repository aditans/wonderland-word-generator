"""
models.py - Model architectures for autoregressive next-word prediction:
1. Simple RNN
2. Bidirectional RNN
3. LSTM
4. GRU
All models share identical embedding dimensionality, recurrent capacity, dropout,
and output classification layer for rigorously comparable experimental conditions.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_simple_rnn(vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5):
    """Constructs a Simple (Elman) Recurrent Neural Network model."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False, name="embedding"),
        layers.SimpleRNN(rnn_units, dropout=dropout_rate, name="simple_rnn"),
        layers.Dense(vocab_size, activation="softmax", name="output_dense")
    ], name="Simple_RNN")
    return model


def build_bidirectional_rnn(vocab_size, embedding_dim=128, rnn_units=64, dropout_rate=0.2, seq_length=5):
    """
    Constructs a Bidirectional Simple RNN model.
    Using 64 forward + 64 backward units = 128 combined representations,
    matching the 128-dimensional hidden representation of unidirectional baselines.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False, name="embedding"),
        layers.Bidirectional(layers.SimpleRNN(rnn_units, dropout=dropout_rate), name="bidirectional_rnn"),
        layers.Dense(vocab_size, activation="softmax", name="output_dense")
    ], name="Bidirectional_RNN")
    return model


def build_lstm(vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5):
    """Constructs a Long Short-Term Memory (LSTM) model with input/forget/output gates."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False, name="embedding"),
        layers.LSTM(rnn_units, dropout=dropout_rate, name="lstm"),
        layers.Dense(vocab_size, activation="softmax", name="output_dense")
    ], name="LSTM")
    return model


def build_gru(vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5):
    """Constructs a Gated Recurrent Unit (GRU) model with update and reset gates."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False, name="embedding"),
        layers.GRU(rnn_units, dropout=dropout_rate, name="gru"),
        layers.Dense(vocab_size, activation="softmax", name="output_dense")
    ], name="GRU")
    return model


def get_all_models(vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5):
    """Factory helper returning all 4 models configured under comparable conditions."""
    return {
        "Simple RNN": build_simple_rnn(vocab_size, embedding_dim, rnn_units, dropout_rate, seq_length),
        "Bidirectional RNN": build_bidirectional_rnn(vocab_size, embedding_dim, rnn_units // 2, dropout_rate, seq_length),
        "LSTM": build_lstm(vocab_size, embedding_dim, rnn_units, dropout_rate, seq_length),
        "GRU": build_gru(vocab_size, embedding_dim, rnn_units, dropout_rate, seq_length)
    }


if __name__ == "__main__":
    test_models = get_all_models(vocab_size=1500)
    for name, m in test_models.items():
        print(f"\n--- {name} (Params: {m.count_params():,}) ---")
        m.summary()
