import pandas as pd
from collections import Counter

# Use the results file you already generated
INPUT_CSV = "verdict_experiment_results.csv"  # or verdict_experiment_results_final.csv

def main():
    print(f"Reading existing results from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    total_evaluated = len(df)
    
    # Check if a 'changed' or matching column exists, otherwise calculate it
    if 'changed' in df.columns:
        scheme_changes = df['changed'].sum()
    else:
        # If columns are named differently, adapt accordingly
        scheme_changes = (df['scheme_with_verdict'] != df['scheme_without_verdict']).sum()

    change_percentage = (scheme_changes / total_evaluated) * 100 if total_evaluated > 0 else 0

    print("\n========================================")
    print("FINAL EXPERIMENT SUMMARY (FROM EXISTING DATA)")
    print("========================================")
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Scheme Changes: {scheme_changes} ({change_percentage:.2f}%)")

    # Generate Scheme Distribution Stats
    scheme_column = 'scheme_with_verdict' if 'scheme_with_verdict' in df.columns else df.columns[1]
    scheme_counts = Counter(df[scheme_column])
    
    print(f"\n--- Top Scheme Predictions (With Verdict) ---")
    for scheme, count in scheme_counts.most_common():
        pct = (count / total_evaluated) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
