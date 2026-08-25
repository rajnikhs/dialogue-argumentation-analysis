import pandas as pd
from collections import Counter
import re

INPUT_CSV = "inferred_schemes_full_output.csv"
OUTPUT_CSV = "clean_scheme_summary_table.csv"

def extract_scheme(text):
    if pd.isna(text):
        return "Unknown"
    # Look for "Scheme: [Name]" pattern in the text block
    match = re.search(r"Scheme:\s*([^\n\r]+)", str(text))
    if match:
        return match.group(1).strip()
    return "Other / Unmatched"

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # Extract clean scheme names
    if 'inferred_scheme_output' in df.columns:
        df['clean_scheme'] = df['inferred_scheme_output'].apply(extract_scheme)
    else:
        print("Error: 'inferred_scheme_output' column not found.")
        return

    # Count distributions
    scheme_counts = Counter(df['clean_scheme'])

    print("\n========================================")
    print("CLEAN ARGUMENT SCHEME DISTRIBUTION")
    print("========================================")
    
    summary_rows = []
    for scheme, count in scheme_counts.most_common():
        pct = (count / total_rows) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")
        summary_rows.append({
            "Argumentation Scheme": scheme,
            "Count": count,
            "Percentage (%)": f"{pct:.2f}%"
        })

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved clean summary table to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
