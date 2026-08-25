import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load data and model
df = pd.read_csv("inferred_schemes_full_output.csv")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

def infer_scheme(text):
    prompt = f"""Analyze the following text snippet from a Reddit dialogue and classify it into one of Douglas Walton's argumentation schemes.
Provide the name of the scheme and a brief rationale.

Text: "{text}"
Format your response as:
Scheme: [Name of Scheme]
Rationale: [Brief explanation]"""

    messages = [
        {"role": "system", "content": "You are an expert in argumentation theory and Walton's argumentation schemes."},
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
    
    # Parse scheme out
    match = re.search(r"Scheme:\s*(.*?)(?:\n|$)", response)
    return match.group(1).strip() if match else "Unparsed"

# 2. Select a subset of 50 posts that actually contain verdict-like terms or just take 50 random samples
subset = df.head(50).copy()

verdict_pattern = re.compile(r'\b(YTA|NTA|ESH|NAH|asshole|ass-hole)\b', re.IGNORECASE)

results = []
print("Starting verdict-masking experiment on 50 posts...")

for idx, row in subset.iterrows():
    original_text = row['displayed_text']
    original_scheme = row['parsed_scheme'] if 'parsed_scheme' in row else "Unknown"
    
    # Strip out verdict words
    masked_text = verdict_pattern.sub("[VERDICT REMOVED]", original_text)
    
    # Run inference on masked text
    try:
        new_scheme = infer_scheme(masked_text)
    except Exception as e:
        new_scheme = "Error"
        
    results.append({
        'instance_id': row.get('instance_id', idx),
        'original_text': original_text,
        'masked_text': masked_text,
        'original_scheme': original_scheme,
        'masked_scheme': new_scheme,
        'changed': original_scheme != new_scheme
    })
    print(f"Row {idx+1}/50 processed. Changed? {original_scheme != new_scheme}")

res_df = pd.DataFrame(results)
res_df.to_csv("verdict_experiment_results.csv", index=False)

# Calculate percentage change
change_count = res_df['changed'].sum()
change_percentage = (change_count / len(res_df)) * 100
print(f"\nExperiment Complete! Scheme changed in {change_count} out of {len(res_df)} posts ({change_percentage}%).")
