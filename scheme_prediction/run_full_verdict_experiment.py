import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

# 1. Load dataset dynamically
csvs = [f for f in glob.glob("*.csv") if "result" not in f and "output" not in f and "statistics" not in f]
print(f"Found dataset files: {csvs}")
target_csv = csvs[0] if csvs else "inferred_schemes_full_output.csv"
df = pd.read_csv(target_csv)
print(f"Loaded {len(df)} rows from {target_csv}")

# 2. Load Model
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
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    # Robust parsing
    match = re.search(r"(?:Scheme:)?\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
    scheme = match.group(1).strip() if match else response.split('\n')[0].strip()
    scheme = scheme.replace("[", "").replace("]", "").replace("*", "").strip()
    return scheme

# Identify text column
text_col = 'displayed_text' if 'displayed_text' in df.columns else [c for c in df.columns if 'text' in c][0]
verdict_pattern = re.compile(r'\b(YTA|NTA|ESH|NAH|asshole|ass-hole)\b', re.IGNORECASE)

results = []
print(f"Starting full experiment across {len(df)} records...")

for idx, row in df.iterrows():
    original_text = row[text_col]
    masked_text = verdict_pattern.sub("[VERDICT REMOVED]", original_text)
    
    orig_scheme = infer_scheme(original_text)
    masked_scheme = infer_scheme(masked_text)
    
    # Normalize for comparison
    norm_orig = orig_scheme.lower().replace(" attack", "").replace("the ", "").strip()
    norm_mask = masked_scheme.lower().replace(" attack", "").replace("the ", "").strip()
    changed = norm_orig != norm_mask
    
    results.append({
        'instance_id': idx,
        'original_text': original_text,
        'masked_text': masked_text,
        'original_scheme': orig_scheme,
        'masked_scheme': masked_scheme,
        'changed': changed
    })
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
        print(f"Processed {idx + 1}/{len(df)} rows...")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("verdict_experiment_results_final.csv", index=False)

# Summary statistics
total = len(results_df)
changes = results_df['changed'].sum()
print("\n" + "="*40)
print("FINAL EXPERIMENT SUMMARY")
print("="*40)
print(f"Total Evaluated: {total}")
print(f"Scheme Changes : {changes} ({(changes/total)*100:.2f}%)")
print("Saved results to verdict_experiment_results_final.csv")
