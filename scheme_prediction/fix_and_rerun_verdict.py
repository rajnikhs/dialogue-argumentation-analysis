import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load data and model
try:
    df_full = pd.read_csv("inferred_schemes_full_output.csv")
except Exception as e:
    print(f"Error loading full output: {e}")
    exit(1)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

def infer_scheme(text):
    prompt = f"""Analyze the following text snippet and classify it into one of Douglas Walton's argumentation schemes.
Provide the name of the scheme and a brief rationale.

Text: "{text}"
Format your response as:
Scheme: [Name of Scheme]
Rationale: [Brief explanation]"""

    messages = [
        {"role": "system", "content": "You are an expert in argumentation theory."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    # Flexible parsing: look for "Scheme:" or take the first line if formatting varies
    match = re.search(r"(?:Scheme:)?\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
    scheme = match.group(1).strip() if match else response.split('\n')[0].strip()
    scheme = scheme.replace("[", "").replace("]", "").replace("*", "").strip()
    return scheme, response

# Select a subset of 50 posts (change to 100 if desired)
subset = df_full.head(50).copy() 
verdict_pattern = re.compile(r'\b(YTA|NTA|ESH|NAH|asshole|ass-hole)\b', re.IGNORECASE)

results = []
print(f"Running verdict-masking experiment on {len(subset)} posts...")

for idx, row in subset.iterrows():
    text_col = 'displayed_text' if 'displayed_text' in row else [c for c in row.index if 'text' in c][0]
    original_text = row[text_col]

    masked_text = verdict_pattern.sub("[VERDICT REMOVED]", original_text)

    print(f"Processing post {idx+1}/{len(subset)}...")
    orig_scheme, orig_resp = infer_scheme(original_text)
    mask_scheme, mask_resp = infer_scheme(masked_text)

    changed = (orig_scheme.lower() != mask_scheme.lower())

    results.append({
        "index": idx,
        "scheme_with_verdict": orig_scheme,
        "scheme_without_verdict": mask_scheme,
        "changed": changed,
        "rationale_with": orig_resp,
        "rationale_without": mask_resp
    })

# Save to final CSV file
df_results = pd.DataFrame(results)
df_results.to_csv("verdict_experiment_results_final.csv", index=False)
print("Experiment complete! Saved results to verdict_experiment_results_final.csv")

# Print percentage change statistics
num_changed = df_results['changed'].sum()
pct_changed = (num_changed / len(df_results)) * 100
print(f"Total Changed: {num_changed}/{len(df_results)} ({pct_changed:.2f}%)")
