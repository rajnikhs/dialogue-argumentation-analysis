import os
os.environ["TRITON_INTERPRET"] = "1"  # Bypasses Triton C-compilation errors (missing Python.h)

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("--- Starting Full Argumentation Scheme Inference Pipeline ---")

# 1. Load the dataset
df = pd.read_csv("fixed_support_attack_dataset.csv")
print(f"Loaded dataset with {len(df)} rows.")

# 2. Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"Loading tokenizer and model: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

print(f"Model loaded and device set to: {model.device}")

# 3. Define the inference function with a few-shot structure
def infer_scheme(text):
    prompt = f"""You are an expert in argumentation mining. Your task is to classify the argumentative input into one of the 22 different argumentation scheme classes. 

Allowed classes: Direct Ad Hominem, Inconsistent Commitment, Cause To Effect, Established Rule, Verbal Classification, Analogy, Example, Precedent, Best Explanation, Ignorance, Sign, Popular Opinion, Popular Practice, Expert Opinion, Position To Know, Witness Testimony, Consequences, Practical Reasoning, Sunk Costs, Threat, Waste, Slippery Slope.

Here are a few examples of how to classify dialogue inputs using these schemes:

Example 1:
ARGUMENT: "You're only saying that because you hate taxes, which proves you're completely biased and untrustworthy."
SCHEME: Direct Ad Hominem
EXPLANATION: The post attacks the arguer's personal character and bias rather than addressing the argument itself.

Example 2:
ARGUMENT: "If we start making exceptions for one person, it will lead down a slippery slope where everyone breaks the rules and total chaos ensues."
SCHEME: Slippery Slope
EXPLANATION: The argument claims that an initial step will trigger a chain reaction resulting in a disastrous final outcome.

---

Now, analyze the following target argument and classify it into one of the 22 classes.
In your answer, provide first the name of the scheme in your output as follows **scheme name**, followed by a short explanation supporting this decision. 

ARGUMENT: "{text}"
"""

    messages = [
        {"role": "system", "content": "You are an expert in argumentation theory and formal scheme classification."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response

# 4. Run inference across the dataset
print("Running inference across the dataset...")
schemes = []
for idx, row in df.iterrows():
    text = row['displayed_text']

    # Handle missing or non-string text gracefully
    if pd.isna(text) or not isinstance(text, str):
        schemes.append("N/A")
        continue

    print(f"Processing row {idx+1}/{len(df)}...")
    try:
        result = infer_scheme(text)
        schemes.append(result)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        schemes.append("Error")

# 5. Save results
df['inferred_scheme_output'] = schemes
output_file = "inferred_schemes_full_output.csv"
df.to_csv(output_file, index=False)
print(f"--- Pipeline Complete! Saved results to {output_file} ---")
