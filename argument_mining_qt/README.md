# QT-Schemes Argument Mining Pipeline

## Overview
This directory contains the code and evaluation outputs for classifying argumentation schemes on the **QT-Schemes** dataset using an `outlines`-constrained `Llama-3-8B-Instruct` pipeline with a comprehensive few-shot definition prompt.

## Contents
- `run_qt_exact.py`: The core inference script using strict `Literal` constraints (26 classes).
- `qt_inferred_schemes_full_output.csv`: Full inference output containing texts, gold labels, and model predictions.
- `evaluate_results.py` / `evaluate_examples.py`: Evaluation scripts checking overall accuracy, classification reports, and behavior on anchor examples.
