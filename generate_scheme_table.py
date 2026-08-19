import pandas as pd

# Load your full results or original scheme output file
df = pd.read_csv("verdict_experiment_results_final.csv")

# Clean up scheme names for accurate counting
scheme_counts = df['original_scheme'].value_counts().reset_index()
scheme_counts.columns = ['Argumentation Scheme', 'Prediction Count']
scheme_counts['Percentage (%)'] = (scheme_counts['Prediction Count'] / len(df)) * 100

print("=== TABLE OF PREDICTIONS PER SCHEME ===")
print(scheme_counts.to_string(index=False))

# Optionally save to CSV for inclusion in a paper/report
scheme_counts.to_csv("scheme_prediction_table.csv", index=False)
