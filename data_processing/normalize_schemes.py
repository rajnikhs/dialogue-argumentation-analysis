import pandas as pd
from collections import Counter
import re

INPUT_CSV = "inferred_schemes_full_output.csv"
OUTPUT_NORMALIZED = "normalized_scheme_summary.csv"

def normalize_scheme_name(raw_text):
    if pd.isna(raw_text):
        return "Unknown"

    text_str = str(raw_text).strip()

    # Extract scheme name using regex (handling markdown asterisks like **Scheme Name**)
    match = re.search(r"\*\*(.*?)\*\*", text_str)
    scheme = match.group(1).strip() if match else text_str

    # Clean up common variations and normalize to lowercase core categories (as suggested)
    s_lower = scheme.lower()

    if "ad hominem" in s_lower:
        return "ad hominem"
    elif "consequence" in s_lower:
        return "appeal to consequences"
    elif "expert" in s_lower or "verecundiam" in s_lower or "authority" in s_lower:
        return "appeal to expert opinion / authority"
    elif "emotion" in s_lower or "pity" in s_lower or "fear" in s_lower or "passiones" in s_lower:
        return "appeal to emotion"
    elif "consistency" in s_lower or "inconsistency" in s_lower:
        return "appeal to consistency"
    elif "analogy" in s_lower or "similarity" in s_lower:
        return "argument from analogy / similarity"
    elif "straw man" in s_lower:
        return "the straw man fallacy"
    elif "ignorance" in s_lower or "ignorantiam" in s_lower:
        return "appeal to ignorance"
    elif "explanation" in s_lower:
        return "appeal to the best explanation"
    elif "tradition" in s_lower or "common practice" in s_lower or "norm" in s_lower:
        return "appeal to tradition / social norms"
    elif "shared human experience" in s_lower or "shared experience" in s_lower:
        return "appeal to shared human experience"
    elif "practical reasoning" in s_lower:
        return "practical reasoning"
    elif "popular opinion" in s_lower:
        return "popular opinion"
    elif "established rule" in s_lower:
        return "established rule"
    elif "slippery slope" in s_lower:
        return "slippery slope"
    elif "verbal classification" in s_lower:
        return "verbal classification"
    elif "popular practice" in s_lower:
        return "popular practice"
    elif "inconsistent commitment" in s_lower:
        return "inconsistent commitment"
    elif "threat" in s_lower:
        return "threat"
    elif "waste" in s_lower:
        return "waste"
    elif "sunk cost" in s_lower:
        return "sunk costs"
    elif "position to know" in s_lower:
        return "position to know"
    elif "cause to effect" in s_lower:
        return "cause to effect"

    # Return lowercase version for any other unique ones
    return s_lower

def main():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)

    # Apply normalization
    df['normalized_scheme'] = df['inferred_scheme_output'].apply(normalize_scheme_name)

    # Group by the normalized scheme and sum counts accurately
    counts = df['normalized_scheme'].value_counts()

    print("\n========================================")
    print("NORMALIZED ARGUMENT SCHEME DISTRIBUTION")
    print("========================================")

    summary_rows = []
    for scheme, count in counts.items():
        pct = (count / total_rows) * 100
        print(f"{scheme}: {count} ({pct:.2f}%)")
        summary_rows.append({
            "Argumentation Scheme": scheme,
            "Count": count,
            "Percentage (%)": f"{pct:.2f}%"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_NORMALIZED, index=False)
    print(f"\nSaved consolidated summary table to {OUTPUT_NORMALIZED}")

if __name__ == "__main__":
    main()
