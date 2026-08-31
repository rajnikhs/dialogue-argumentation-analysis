import os
os.environ["TRITON_INTERPRET"] = "1"

import pandas as pd
import torch
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines

print("--- Starting Outlines-Constrained Argumentation Scheme Inference Pipeline ---")

# 1. Define the exact 22 valid schemes using Literal (Outlines v1.3.3 native choice constraint)
ValidSchemes = Literal[
    "Direct Ad Hominem",
    "Inconsistent Commitment",
    "Cause To Effect",
    "Established Rule",
    "Verbal Classification",
    "Analogy",
    "Example",
    "Precedent",
    "Best Explanation",
    "Ignorance",
    "Sign",
    "Popular Opinion",
    "Popular Practice",
    "Expert Opinion",
    "Position To Know",
    "Witness Testimony",
    "Consequences",
    "Practical Reasoning",
    "Sunk Costs",
    "Threat",
    "Waste",
    "Slippery Slope"
]

# 2. Load the dataset
df = pd.read_csv("fixed_support_attack_dataset.csv")
print(f"Loaded dataset with {len(df)} rows.")

# 3. Load model using outlines.from_transformers wrapper
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"Loading and wrapping model: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

# Initialize outlines model instance using from_transformers
model = outlines.from_transformers(hf_model, tokenizer)

# 4. Define the inference function using Outlines type-driven API
def infer_scheme(text):
    prompt = f"""Classify the following argument into one of the 22 valid argumentation schemes.

ARGUMENT:
"{text}"

Scheme:"""
    
    # Passing the Literal type directly to the model enforces strict choice generation
    predicted_scheme = model(prompt, ValidSchemes)
    return predicted_scheme

# 5. Run inference across the dataset
print("Running constrained inference across the dataset...")
schemes = []
for idx, row in df.iterrows():
    text = row['displayed_text']

    if pd.isna(text) or not isinstance(text, str):
        schemes.append("N/A")
        continue

    print(f"Processing row {idx+1}/{len(df)}...")
    try:
        result = infer_scheme(text)
        schemes.append(str(result))
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        schemes.append("Error")

# 6. Save results
df['inferred_scheme_output'] = schemes
output_file = "inferred_schemes_full_output.csv"
df.to_csv(output_file, index=False)
print(f"--- Pipeline Complete! Saved strictly constrained results to {output_file} ---")
