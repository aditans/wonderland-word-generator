"""
evaluate.py - Quantitative evaluation and visualization module.
Plots loss curves, accuracy curves, perplexity comparisons, training times,
and qualitative generation analyses.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join("results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling palette
COLORS = {
    "Simple RNN": "#e74c3c",       # Crimson Red
    "Bidirectional RNN": "#e67e22", # Amber Orange
    "LSTM": "#2980b9",              # Royal Blue
    "GRU": "#27ae60"                # Emerald Green
}


def plot_loss_and_accuracy(history_dict, save_dir=PLOTS_DIR):
    """Generates two separate dual-axis / comparative plots for Loss and Accuracy trajectories."""
    # 1. Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in history_dict.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=f"{name} (Train)", color=COLORS.get(name, "#333"), linestyle="--", alpha=0.7)
        axes[0].plot(epochs, hist["val_loss"], label=f"{name} (Val)", color=COLORS.get(name, "#333"), linewidth=2)

    axes[0].set_title("Training & Validation Loss Across Epochs", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9, loc="upper right")

    # 2. Accuracy Curves
    for name, hist in history_dict.items():
        epochs = range(1, len(hist["train_acc"]) + 1)
        train_acc_pct = [a * 100 for a in hist["train_acc"]]
        val_acc_pct = [a * 100 for a in hist["val_acc"]]
        axes[1].plot(epochs, train_acc_pct, label=f"{name} (Train)", color=COLORS.get(name, "#333"), linestyle="--", alpha=0.7)
        axes[1].plot(epochs, val_acc_pct, label=f"{name} (Val)", color=COLORS.get(name, "#333"), linewidth=2)

    axes[1].set_title("Training & Validation Accuracy (%) Across Epochs", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    loss_acc_path = os.path.join(save_dir, "loss_and_accuracy_curves.png")
    plt.savefig(loss_acc_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved loss & accuracy curves to {loss_acc_path}")


def plot_metrics_bars(summary_list, save_dir=PLOTS_DIR):
    """Generates comparison bar charts for Perplexity and Training Time."""
    df = pd.DataFrame(summary_list)
    names = df["model_name"].tolist()
    colors = [COLORS.get(n, "#555") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Perplexity comparison (Lower is better)
    bars1 = axes[0].bar(names, df["final_val_ppl"], color=colors, alpha=0.85, edgecolor="black", width=0.55)
    axes[0].set_title("Validation Perplexity (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Perplexity (exp(Loss))", fontsize=11)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Training time comparison
    bars2 = axes[1].bar(names, df["total_training_time_sec"], color=colors, alpha=0.85, edgecolor="black", width=0.55)
    axes[1].set_title("Total Training Runtime (Seconds)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Time (seconds)", fontsize=11)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    metrics_path = os.path.join(save_dir, "perplexity_and_runtime_comparison.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved perplexity & runtime comparison to {metrics_path}")


def plot_temperature_effects(generation_results, save_dir=PLOTS_DIR):
    """Visualizes average diversity and repetition rates across temperature values."""
    temps = [0.5, 0.8, 1.2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, temp_dict in generation_results.items():
        avg_divs = []
        avg_reps = []
        for t in temps:
            samples = temp_dict.get(str(t), [])
            div = np.mean([s["diversity"] for s in samples]) if samples else 0
            rep = np.mean([s["repetition_rate"] for s in samples]) if samples else 0
            avg_divs.append(div)
            avg_reps.append(rep)

        axes[0].plot(temps, avg_divs, marker="o", linewidth=2, label=model_name, color=COLORS.get(model_name, "#333"))
        axes[1].plot(temps, avg_reps, marker="s", linewidth=2, label=model_name, color=COLORS.get(model_name, "#333"))

    axes[0].set_title("Vocabulary Diversity vs Sampling Temperature", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Sampling Temperature (T)", fontsize=11)
    axes[0].set_ylabel("Diversity Score (Unique / Total Words)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Repetition Rate vs Sampling Temperature", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Sampling Temperature (T)", fontsize=11)
    axes[1].set_ylabel("Repetition Rate (Duplicated Bigrams)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    temp_path = os.path.join(save_dir, "temperature_effect_analysis.png")
    plt.savefig(temp_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved temperature effect plots to {temp_path}")
