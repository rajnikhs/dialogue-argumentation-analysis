import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load the output file
df = pd.read_csv("qt_inferred_schemes_full_output.csv")

# Clean and normalize gold labels and predictions for fair comparison
df['scheme_gold_clean'] = df['scheme_gold'].dropna().str.strip().str.title()
df['scheme_pred_clean'] = df['inferred_scheme_output'].dropna().str.strip().str.title()

# Filter out rows where we had errors or N/A
valid_df = df.dropna(subset=['scheme_gold_clean', 'scheme_pred_clean'])
valid_df = valid_df[valid_df['scheme_pred_clean'] != 'Error']

print(f"Total rows evaluated: {len(valid_df)} out of {len(df)}")

# Calculate accuracy
acc = accuracy_score(valid_df['scheme_gold_clean'], valid_df['scheme_pred_clean'])
print(f"Accuracy: {acc:.4f}\n")

# Print full classification report
print("Classification Report:")
print(classification_report(
    valid_df['scheme_gold_clean'], 
    valid_df['scheme_pred_clean'], 
    zero_division=0
))
