import pandas as pd

# Load the experiment results
df = pd.read_csv("verdict_experiment_results_final.csv")

# Filter out minor parsing artifacts (like literal text snippets captured as schemes)
df_clean = df[~df['original_scheme'].str.startswith('Text:', na=False)]
df_clean = df_clean[~df_clean['masked_scheme'].str.startswith('Text:', na=False)]

print(f"Total rows analyzed after cleaning artifacts: {len(df_clean)}")

# 1. Top Category Shifts (Cross-tabulation / Transition Matrix)
transition_matrix = pd.crosstab(
    df_clean['original_scheme'], 
    df_clean['masked_scheme'], 
    rownames=['Original Scheme'], 
    colnames=['Masked Scheme']
)

print("\n=== TOP SCHEME TRANSITIONS (Full Matrix Summary) ===")
# Show only rows/cols that actually changed or have values to keep it readable
print(transition_matrix)

# 2. Most Frequent Specific Shifts
print("\n=== MOST COMMON SPECIFIC MIGRATIONS ===")
shifts = df_clean[df_clean['original_scheme'] != df_clean['masked_scheme']]
common_shifts = shifts.groupby(['original_scheme', 'masked_scheme']).size().reset_index(name='count')
common_shifts = common_shifts.sort_values(by='count', ascending=False)

print(common_shifts.head(10).to_string(index=False))
