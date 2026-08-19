import pandas as pd
from collections import Counter

INPUT_CSV = "verdict_experiment_results.csv"
OUTPUT_TABLE = "scheme_prediction_table.csv"

def main():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    total_evaluated = len(df)
    print(f"Total Rows Evaluated: {total_evaluated}")
    
    # 1. Calculate Shift Percentage
    if 'changed' in df.columns:
        # If 'changed' is boolean or 0/1
        scheme_changes = df['changed'].sum()
    else:
        scheme_changes = (df['original_scheme'] != df['masked_scheme']).sum()
        
    change_percentage = (scheme_changes / total_evaluated) * 100 if total_evaluated > 0 else 0

    print("\n========================================")
    print("FINAL EXPERIMENT SUMMARY METRICS")
    print("========================================")
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Scheme Changes: {scheme_changes} ({change_percentage:.2f}%)")

    # 2. Get Distribution of Schemes (With Verdict)
    print("\n--- Argument Scheme Distribution (With Verdict) ---")
    orig_counts = Counter(df['original_scheme'])
    for scheme, count in orig_counts.most_common():
        pct = (count / total_evaluated) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")

    # 3. Get Distribution of Schemes (Without Verdict / Masked)
    print("\n--- Argument Scheme Distribution (Without Verdict) ---")
    masked_counts = Counter(df['masked_scheme'])
    for scheme, count in masked_counts.most_common():
        pct = (count / total_evaluated) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")

    # 4. Create and Save the Final Table
    summary_rows = []
    all_schemes = set(list(orig_counts.keys()) + list(masked_counts.keys()))
    
    for scheme in sorted(all_schemes):
        with_count = orig_counts.get(scheme, 0)
        without_count = masked_counts.get(scheme, 0)
        with_pct = (with_count / total_evaluated) * 100
        without_pct = (without_count / total_evaluated) * 100
        
        summary_rows.append({
            "Argumentation Scheme": scheme,
            "# Predictions (With Verdict)": with_count,
            "% of Total (With)": f"{with_pct:.2f}%",
            "# Predictions (Without Verdict)": without_count,
            "% of Total (Without)": f"{without_pct:.2f}%"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"\nSuccessfully generated and saved scheme prediction table to {OUTPUT_TABLE}")
    print("\nPreview of Summary Table:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
