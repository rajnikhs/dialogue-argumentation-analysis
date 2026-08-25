import pandas as pd

# Load the experiment results
df = pd.read_csv("verdict_experiment_results.csv")

# Function to normalize scheme names for fair comparison
def normalize_scheme(name):
    if not isinstance(name, str):
        return "unknown"
    # Lowercase, strip whitespace, remove common noise words like "attack", "scheme", etc.
    clean = name.lower().strip()
    clean = clean.replace(" attack", "").replace("the ", "").replace("scheme", "").strip()
    return clean

# Apply normalization
df['norm_original'] = df['original_scheme'].apply(normalize_scheme)
df['norm_masked'] = df['masked_scheme'].apply(normalize_scheme)

# Re-evaluate change with normalization
df['true_change'] = df['norm_original'] != df['norm_masked']

true_change_count = df['true_change'].sum()
true_change_percentage = (true_change_count / len(df)) * 100

print(f"=== VERDICT EXPERIMENT SUMMARY (Normalized) ===")
print(f"Total Posts Tested: {len(df)}")
print(f"True Conceptual Scheme Changes: {true_change_count} ({true_change_percentage:.2f}%)\n")

print("=== SIDE-BY-SIDE INSPECTION (First 5 Rows) ===")
for idx, row in df.head(5).iterrows():
    print(f"[{idx+1}] Text Snippet: {row['original_text'][:120]}...")
    print(f"   Original Scheme : {row['original_scheme']} --> Normalized: {row['norm_original']}")
    print(f"   Masked Scheme   : {row['masked_scheme']} --> Normalized: {row['norm_masked']}")
    print(f"   Did it change?  : {row['true_change']}")
    print("-" * 70)
