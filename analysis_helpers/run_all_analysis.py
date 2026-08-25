import os
import pandas as pd
from collections import Counter

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
INPUT_CSV = "your_input_dataset.csv"  # Update with your actual dataset path
OUTPUT_CSV = "verdict_experiment_results_final.csv"

def query_llm_for_scheme(post_text):
    """
    TODO: Drop your actual model inference logic here. 
    This function should take the post text (with or without verdict) 
    and return a tuple: (predicted_scheme, rationale)
    
    Example:
        response = client.chat.completions.create(...)
        scheme = extract_scheme(response)
        rationale = extract_rationale(response)
        return scheme, rationale
    """
    # Replace this mock block with your real implementation:
    raise NotImplementedError("Please plug in your actual model inference code here!")

def run_experiment(df):
    total_evaluated = 0
    scheme_changes = 0
    results = []

    print(f"Starting analysis on {len(df)} posts...")

    for idx, row in df.iterrows():
        total_evaluated += 1
        
        # Pull text components
        # (Modify column names 'post_text' and 'verdict' to match your dataframe)
        original_text = row['post_text'] 
        verdict_text = row['verdict'] 
        
        # 1. Run WITH verdict (Original text)
        scheme_with, rat_with = query_llm_for_scheme(original_text)
        
        # 2. Run WITHOUT verdict (Masked text: strip or remove the verdict string)
        masked_text = original_text.replace(verdict_text, "") # Adjust masking logic if needed
        scheme_without, rat_without = query_llm_for_scheme(masked_text)
        
        # Track changes
        changed = (scheme_with != scheme_without)
        if changed:
            scheme_changes += 1
            
        results.append({
            "index": idx,
            "scheme_with_verdict": scheme_with,
            "scheme_without_verdict": scheme_without,
            "changed": changed,
            "rationale_with": rat_with,
            "rationale_without": rat_without
        })
        
        if total_evaluated % 100 == 0:
            print(f"Processed {total_evaluated}/{len(df)} posts...")

    return pd.DataFrame(results), total_evaluated, scheme_changes

def main():
    print("Loading dataset...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find dataset at {INPUT_CSV}. Please update INPUT_CSV path.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Run pipeline
    results_df, total_evaluated, scheme_changes = run_experiment(df)
    
    # Save results
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    # Compute final metrics
    change_percentage = (scheme_changes / total_evaluated) * 100 if total_evaluated > 0 else 0

    print("\n========================================")
    print("FINAL EXPERIMENT SUMMARY")
    print("========================================")
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Scheme Changes: {scheme_changes} ({change_percentage:.2f}%)")
    print(f"Saved results to {OUTPUT_CSV}")

    # Generate Scheme Distribution Stats
    scheme_counts = Counter(results_df['scheme_with_verdict'])
    print("\n--- Top Scheme Predictions (With Verdict) ---")
    for scheme, count in scheme_counts.most_common():
        pct = (count / total_evaluated) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
