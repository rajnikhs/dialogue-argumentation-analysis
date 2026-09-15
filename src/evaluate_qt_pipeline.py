import os
os.environ["TRITON_INTERPRET"] = "1"

import pandas as pd
import torch
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines

print("--- Initializing Llama-3 for QT-Schemes Diagnostic Evaluation ---")

# Define the exact 26 normalized schemes from the QT-Schemes dataset as a Literal type
ValidQTSchemes = Literal[
    'Allegation Of Bias',
    'Analogy',
    'Best Explanation',
    'Cause To Effect',
    'Consequences',
    'Default Inference',
    'Default Inference (Qa)',
    'Direct Ad Hominem',
    'Established Rule',
    'Example',
    'Expert Opinion',
    'Ignorance',
    'Inconsistent Commitment',
    'Popular Opinion',
    'Popular Practice',
    'Position To Know',
    'Practical Reasoning',
    'Precedent',
    'Random Sample To Population',
    'Sign',
    'Slippery Slope',
    'Sunk Costs',
    'Threat',
    'Verbal Classification',
    'Waste',
    'Witness Testimony'
]

# Load model and tokenizer (adjusting paths to match your typical environment setup if needed)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize outlines generator
generator = outlines.generate.choice(model, ValidQTSchemes)

# Load data
df = pd.read_csv('qt_schemes_repo/QT-Schemes/10June2021.csv', sep=';')
df['scheme_normalized'] = df['scheme_gold'].dropna().str.strip().str.title()
sample_df = df.head(5)

print("\n--- Running Inference on Sample Arguments ---")
for idx, row in sample_df.iterrows():
    text = row['text']
    gold = row['scheme_normalized']
    
    prompt = f"Analyze the following argument and classify its argumentation scheme:\n\nArgument: {text}\n\nScheme:"
    
    # Generate constrained response
    prediction = generator(prompt)
    
    print(f"\n[Argument]: {text[:80]}...")
    print(f" -> Gold Scheme: {gold}")
    print(f" -> Llama Prediction: {prediction}")

print("\nDiagnostic evaluation complete!")
