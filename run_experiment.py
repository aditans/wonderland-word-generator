"""
run_experiment.py - Master Orchestrator for Assignment 2.
1. Prepares dataset from Project Gutenberg.
2. Trains Simple RNN, Bidirectional RNN, LSTM, and GRU under controlled conditions.
3. Performs temperature-scaled autoregressive text generation on 4 seed phrases (T=0.5, 0.8, 1.2).
4. Plots quantitative trajectories (loss, accuracy, perplexity, time) & qualitative trends.
5. Generates the submission Word document and PDF.
"""

import sys
import os
import json
import pandas as pd

# Add src to python path
sys.path.append(os.path.abspath("src"))

from dataset import prepare_dataset
from train import run_training_pipeline
from generate import evaluate_qualitative_generation
from evaluate import plot_loss_and_accuracy, plot_metrics_bars, plot_temperature_effects
from generate_submission_doc import create_submission_document, convert_docx_to_pdf


def main():
    print("=====================================================================")
    print("  FOUNDATIONS OF GENERATIVE AI (ICT4403) - ASSIGNMENT 2 PIPELINE     ")
    print("  Project: Wonderland Word Generator                                ")
    print("=====================================================================\n")

    # 1. Train all 4 models under identical conditions
    # 10 epochs gives strong convergence while maintaining fast execution
    trained_models, all_histories, all_summaries, word2idx, idx2word = run_training_pipeline(
        epochs=10, batch_size=128, lr=0.003
    )

    # 2. Evaluate qualitative text generation across 3 sampling temperatures (0.5, 0.8, 1.2)
    print("\n[Step 2] Performing Temperature-Scaled Autoregressive Text Generation...")
    gen_results, table_rows = evaluate_qualitative_generation(
        trained_models, word2idx, idx2word,
        temperatures=[0.5, 0.8, 1.2],
        gen_words=20
    )

    # Save readable qualitative report
    report_path = os.path.join("results", "qualitative_generation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("====================================================================================\n")
        f.write("      QUALITATIVE TEXT GENERATION REPORT - WONDERLAND WORD GENERATOR               \n")
        f.write("====================================================================================\n\n")
        for row in table_rows:
            f.write(f"Model: {row['Model']} | Temperature: {row['Temp']}\n")
            f.write(f"Seed: \"{row['Seed']}\"\n")
            f.write(f"Generated: \"{row['Generated Passage']}\"\n")
            f.write(f"Diversity: {row['Diversity']} | Repetition Rate: {row['Repetition Rate']}\n")
            f.write("-" * 80 + "\n")
    print(f"[Step 2] Saved qualitative report to {report_path}")

    # 3. Generate Evaluation Plots
    print("\n[Step 3] Generating Publication-Quality Figures & Evaluation Charts...")
    plot_loss_and_accuracy(all_histories)
    plot_metrics_bars(all_summaries)
    plot_temperature_effects(gen_results)

    # 4. Create Student Submission Document & PDF
    print("\n[Step 4] Building Submission Word Document & PDF...")
    docx_file = create_submission_document()
    convert_docx_to_pdf(docx_file)

    print("\n=====================================================================")
    print("  EXPERIMENT COMPLETE! All models trained, evaluated, and saved.      ")
    print("=====================================================================")


if __name__ == "__main__":
    main()
