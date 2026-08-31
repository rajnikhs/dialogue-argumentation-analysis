import pandas as pd

# Load your file
df = pd.read_csv("inferred_schemes_full_output.csv")

# Find the prediction column
target_col = [c for c in df.columns if "scheme" in c.lower() or "prediction" in c.lower() or "output" in c.lower()][0]

def standardize_scheme(text):
    text_lower = str(text).lower()
    if "expert" in text_lower:
        return "Expert Opinion"
    elif "practical" in text_lower:
        return "Practical Reasoning"
    elif "popular" in text_lower or "consensus" in text_lower:
        return "Popular Opinion"
    elif "analogy" in text_lower:
        return "Analogy"
    elif "consequence" in text_lower:
        return "Consequences"
    elif "commitment" in text_lower:
        return "Commitment"
    # Add any other categories from your paper/prompt here
    else:
        # Fallback: grab the first 3-4 words or truncate
        return str(text)[:40].strip()

df['clean_scheme_name'] = df[target_col].apply(standardize_scheme)

# Save the properly cleaned file
df.to_csv("inferred_schemes_full_output_cleaned.csv", index=False)
print("Done! Clean scheme names successfully mapped.")
