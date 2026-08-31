import pandas as pd

df = pd.read_csv("posts_with_topics.csv")
print(f"Loaded dataset with {len(df)} rows.")

# Use the exact column names present in your dataset
scheme_col = 'scheme_name_only' if 'scheme_name_only' in df.columns else 'inferred_scheme_output'
topic_col = 'topic_name' if 'topic_name' in df.columns else 'dominant_topic'

if scheme_col in df.columns and topic_col in df.columns:
    print(f"Using scheme column: '{scheme_col}' and topic column: '{topic_col}'")
    
    # Generate the crosstab matrix (Topics as rows, Clean Schemes as columns)
    ct = pd.crosstab(df[topic_col], df[scheme_col], margins=True)
    print("\n--- Cross-Tabulation Matrix ---")
    print(ct)
    
    # Save the cross-tabulation table
    ct.to_csv("topic_vs_scheme_crosstab.csv")
    print("\nSaved clean cross-tabulation matrix to topic_vs_scheme_crosstab.csv")
else:
    print("Required columns not found.")
    print("Available columns:", df.columns.tolist())
