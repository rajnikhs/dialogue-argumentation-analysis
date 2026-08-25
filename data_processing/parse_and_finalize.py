import os
import pandas as pd
from collections import Counter
import re

INPUT_CSV = "verdict_experiment_results.csv"
OUTPUT_TABLE = "scheme_prediction_table.csv"

def extract_scheme_name(raw_text):
    """
    Extracts standard scheme name from raw model output text.
    Adjust the regex/matching based on your prompt output format.
    """
    if pd.isna(raw_text):
        return "Unknown"
    
    text_str = str(raw_text).strip()
    
    # Common schemes in your taxonomy to look for
    known_schemes = [
        "Ad Hominem", "Appeal to Consequences", "Appeal to Expert Opinion",
        "Appeal to Popular Opinion", "Argument from Analogy", "Practical Reasoning",
        "Appeal to Ignorance", "Slippery Slope", "False Dilemma"
    ]
    
    for scheme in known_schemes:
        if scheme.lower() in text_str.lower():
            return scheme
            
    # Fallback: take the first sentence or clean chunk if it doesn't match a known keyword
    match = re.split(r'[.\n:]', text_str)
    return match[0].strip() if match else "Other"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Processing data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    total_rows = len(df)
    print(f"Total rows loaded: {total_rows}")

    # Inspect column names to find the right ones
    print("Columns found:", df.columns.tolist())

    # Example processing assuming standard columns exist
    # Let's map or locate the scheme columns
    # (If your columns have different names, update them below)
    
    # For demonstration, let's look for columns containing 'scheme' or text predictions
    scheme_cols = [c for c in df.columns if 'scheme' in c.lower()]
    print(f"Identified scheme-related columns: {scheme_cols}")

if __name__ == "__main__":
    main()
