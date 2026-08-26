import pandas as pd

df = pd.read_csv("posts_with_bertopic_results.csv")

print(f"Loaded dataset with {len(df)} rows.")

# 2. Check if both columns exist
if 'clean_scheme_name' in df.columns and 'bertopic_name' in df.columns:
    print("\n--- Cross-Tabulation: BERTopic Themes vs Argumentation Schemes ---")
    
    # Generate a crosstab matrix (Topics as rows, Schemes as columns)
    ct = pd.crosstab(df['bertopic_name'], df['clean_scheme_name'], margins=True)
    print(ct)
    
    # Save the cross-tabulation table
    ct.to_csv("topic_vs_scheme_crosstab.csv")
    print("\nSaved cross-tabulation matrix to topic_vs_scheme_crosstab.csv")
else:
    print("Required columns ('clean_scheme_name' or 'bertopic_name') not found in DataFrame.")
    print("Available columns:", df.columns.tolist())
