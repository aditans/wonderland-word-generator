"""
create_notebook.py - Programmatically creates the clean, fully-rendered
wonderland_word_generator.ipynb Jupyter Notebook using standard JSON.
"""

import os
import json


def make_md_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    }


def make_code_cell(code_text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_text.strip().split("\n")]
    }


cells = []

# Title & Overview
cells.append(make_md_cell("""# Wonderland Word Generator
### Foundations of Generative AI (ICT4403) — Assignment 2
**Student Name:** Aditanshu Sahu  
**Registration Number:** 230953444  
**GitHub Repository:** [https://github.com/aditans/wonderland-word-generator](https://github.com/aditans/wonderland-word-generator)  
**Dataset:** Lewis Carroll's *Alice's Adventures in Wonderland* ([Project Gutenberg #11](https://www.gutenberg.org/ebooks/11))

---

## 1. Problem Statement & Objectives
The goal of this assignment is to develop an autoregressive, word-level text-generation system using four distinct recurrent neural network architectures:
1. **Simple RNN** (Vanilla Elman RNN)
2. **Bidirectional RNN** (Bidirectional Simple RNN)
3. **LSTM** (Long Short-Term Memory)
4. **GRU** (Gated Recurrent Unit)

### Core Tasks:
- **Task i**: Ingest, clean, tokenize, and construct autoregressive input–target rolling sequences from *Alice's Adventures in Wonderland*. Train all 4 models under rigorously identical experimental conditions.
- **Task ii**: Quantitatively benchmark all architectures on cross-entropy loss, perplexity ($PPL = \\exp(\\mathcal{L})$), next-word accuracy, and computational runtime.
- **Task iii**: Qualitatively evaluate text generation using temperature-scaled multinomial sampling ($T \\in \\{0.5, 0.8, 1.2\\}$) across standardized seed sequences, analyzing the influence of temperature on coherence, diversity, and repetition."""))

# Section 2: Mathematical Formulations
cells.append(make_md_cell("""## 2. Theoretical Architecture & Mathematical Formulations

### 2.1 Simple (Elman) RNN
Computes hidden state transitions via a single hyperbolic tangent nonlinearity:
$$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$\\hat{y}_t = \\text{softmax}(W_{hy} h_t + b_y)$$
*Limitation:* Susceptible to vanishing/exploding gradients over extended contexts.

### 2.2 Bidirectional RNN
Processes the sequence in both forward ($\\overrightarrow{h_t}$) and backward ($\\overleftarrow{h_t}$) temporal directions:
$$\\overrightarrow{h_t} = \\tanh(\\overrightarrow{W}_{xh} x_t + \\overrightarrow{W}_{hh} \\overrightarrow{h}_{t-1} + \\vec{b}_h)$$
$$\\overleftarrow{h_t} = \\tanh(\\overleftarrow{W}_{xh} x_t + \\overleftarrow{W}_{hh} \\overleftarrow{h}_{t+1} + \\overleftarrow{b}_h)$$
$$h_t = [\\overrightarrow{h_t} \\,;\\, \\overleftarrow{h_t}]$$

### 2.3 Long Short-Term Memory (LSTM)
Overcomes the vanishing gradient problem using cell state $c_t$ governed by three interactive multiplicative gates:
- **Forget Gate:** $f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$
- **Input Gate:** $i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$
- **Candidate Memory:** $\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$
- **Cell State Update:** $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$
- **Output Gate:** $o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$
- **Hidden State:** $h_t = o_t \\odot \\tanh(c_t)$

### 2.4 Gated Recurrent Unit (GRU)
A streamlined alternative to LSTM combining forget and input gates into an update gate:
- **Update Gate:** $z_t = \\sigma(W_z [h_{t-1}, x_t] + b_z)$
- **Reset Gate:** $r_t = \\sigma(W_r [h_{t-1}, x_t] + b_r)$
- **Candidate State:** $\\tilde{h}_t = \\tanh(W [r_t \\odot h_{t-1}, x_t] + b)$
- **Hidden State:** $h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$

### 2.5 Sampling Temperature in Autoregressive Generation
To balance greedy determinism and exploratory randomness:
$$P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_{j=1}^{|V|} \\exp(z_j / T)}$$
- $T < 1.0$: Sharpens probability distribution (higher coherence, lower diversity, potential repetition loops).
- $T = 1.0$: Unaltered model logits.
- $T > 1.0$: Flattens distribution (higher entropy, rich vocabulary, risk of syntactic drift)."""))

# Section 3: Imports & Setup
cells.append(make_code_cell("""import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Add src to system path
sys.path.append(os.path.abspath("../src"))
from dataset import prepare_dataset, download_dataset, preprocess_text
from models import get_all_models
from generate import generate_text_sequence, calculate_text_metrics

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")"""))

# Section 4: Dataset Ingestion & Preprocessing
cells.append(make_md_cell("""## 3. Dataset Preprocessing & Sequence Construction
We download *Alice's Adventures in Wonderland* directly from Project Gutenberg, strip administrative boilerplate headers/footers, normalize punctuation and contractions, and construct sliding window sequences of length $L=5$."""))

cells.append(make_code_cell("""# Prepare dataset and examine characteristics
X, y, word2idx, idx2word, metadata = prepare_dataset(seq_length=5, max_vocab_size=3000, min_freq=2)
vocab_size = len(word2idx)

print("Dataset Summary:")
for k, v in metadata.items():
    print(f" - {k}: {v}")

print(f"\\nInput tensor shape: {X.shape}")
print(f"Target tensor shape: {y.shape}")
print(f"Sample sequence: {[idx2word[i] for i in X[0]]} -> Next word: {idx2word[y[0]]}")"""))

# Section 5: Model Architectures
cells.append(make_md_cell("""## 4. Controlled Architectural Implementations
All models share:
- Input shape: $(5,)$
- Embedding dimension: $128$
- Recurrent capacity: $128$ hidden units (64 forward + 64 backward for BiRNN)
- Recurrent dropout: $0.2$
- Softmax output dimension: $|V| = 1,466$"""))

cells.append(make_code_cell("""models_dict = get_all_models(vocab_size=vocab_size, embedding_dim=128, rnn_units=128, dropout_rate=0.2, seq_length=5)
for name, m in models_dict.items():
    print(f"{name:20s} Trainable Parameters: {m.count_params():,}")"""))

# Section 6: Quantitative Comparison
cells.append(make_md_cell("""## 5. Quantitative Evaluation & Benchmark Results
We evaluate the four architectures across Validation Loss, Validation Accuracy, Perplexity ($PPL = \\exp(\\text{Loss})$), and Training Speed."""))

cells.append(make_code_cell("""# Load the actual empirical benchmark metrics recorded during training
metrics_csv_path = os.path.join("..", "results", "metrics_comparison.csv")
df_metrics = pd.read_csv(metrics_csv_path)
display(df_metrics)"""))

# Section 7: Visualizations
cells.append(make_md_cell("""### 5.1 Training Curves & Performance Charts"""))
cells.append(make_code_cell("""# Display loss and accuracy trajectories
display(Image(filename=os.path.join("..", "results", "plots", "loss_and_accuracy_curves.png")))
display(Image(filename=os.path.join("..", "results", "plots", "perplexity_and_runtime_comparison.png")))"""))

# Section 8: Temperature Sampling Analysis
cells.append(make_md_cell("""## 6. Qualitative Sampling Temperature Analysis
We generate text from standardized seeds across three temperatures:
- **$T = 0.5$ (Low):** Deterministic, high local grammatical structure, tendency toward repetitive cycles.
- **$T = 0.8$ (Medium):** Balanced exploration, rich narrative flow, natural vocabulary variety.
- **$T = 1.2$ (High):** Diverse and creative, with elevated entropy and occasional syntactic drift."""))

cells.append(make_code_cell("""display(Image(filename=os.path.join("..", "results", "plots", "temperature_effect_analysis.png")))"""))

cells.append(make_code_cell("""# Load and display qualitative generation samples across all models
samples_json_path = os.path.join("..", "results", "generation_samples.json")
with open(samples_json_path, "r", encoding="utf-8") as f:
    generation_data = json.load(f)

# Display sample comparisons for Seed: 'the cat only grinned when'
seed_target = "the cat only grinned when"
print(f"=== Comparative Generation for Seed: '{seed_target}' ===\\n")
for model_name, temps in generation_data.items():
    print(f"--- {model_name} ---")
    for t_val in ["0.5", "0.8", "1.2"]:
        for sample in temps[t_val]:
            if sample["seed"] == seed_target:
                print(f" [T={t_val}] {sample['full_passage']}")
                print(f"       Diversity: {sample['diversity']} | Repetition: {sample['repetition_rate']}")
    print()"""))

# Section 9: Conclusion
cells.append(make_md_cell("""## 7. Analysis & Conclusion

### Summary of Architectural Insights:
1. **LSTM (Best Overall Performer):**
   - Achieved the lowest validation cross-entropy loss (**4.9833**) and best validation perplexity (**145.96**).
   - The cell state gate design ($f_t, i_t, o_t$) effectively retained context over the rolling window, producing the most syntactically natural, contextually faithful passages.
2. **GRU (Strong Convergence with Minor Overfitting):**
   - Converged most rapidly during training (training loss: **2.1242**, training accuracy: **49.28%**).
   - Slightly higher validation perplexity (**201.10**), showing that its dual-gate mechanism fits rapidly to small corpora but requires slightly higher regularization to match LSTM generalization.
3. **Simple RNN (Fastest Runtime, Lower Generalization):**
   - Fastest training runtime (**21.39s** total, **2.14s/epoch**).
   - Higher validation loss (**5.1427**) and perplexity (**171.18**) due to vanishing gradients in un-gated recurrent transitions.
4. **Bidirectional RNN (Causal Constraint in Autoregression):**
   - Demonstrated high training fit (**41.90%**), but during autoregressive left-to-right inference, the backward pass lacks future context. Hence, causal architectures (LSTM/GRU) demonstrate superior generative stability.

**Final Verdict:** **LSTM** at sampling temperature **$T = 0.8$** delivers the optimal balance of narrative coherence, stylistic authenticity, and lexical diversity for *Alice's Adventures in Wonderland*."""))

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join("notebooks", "wonderland_word_generator.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Successfully generated {out_path}")
