import pandas as pd

df = pd.read_csv("verdict_experiment_results_final.csv")

print("--- Overall Statistics ---")
print(f"Total pairs analyzed: {len(df)}")
print(f"Changes detected: {df['changed'].sum()} ({df['changed'].mean()*100:.2f}%)")

print("\n--- Most Common Original Schemes ---")
print(df['original_scheme'].value_counts().head(5))

print("\n--- Most Common Masked Schemes ---")
print(df['masked_scheme'].value_counts().head(5))

print("\n--- Sample Changes (Where Verdict Masking Altered the Scheme) ---")
changed_df = df[df['changed'] == True]
if not len(changed_df) == 0:
    for idx, row in changed_df.head(3).iterrows():
        print(f"\nOriginal Text: {row['original_text'][:150]}...")
        print(f"  -> Original Scheme: {row['original_scheme']}")
        print(f"  -> Masked Scheme  : {row['masked_scheme']}")
else:
    print("No scheme changes found yet.")
