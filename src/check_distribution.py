import pandas as pd

# Load the output file
df = pd.read_csv("inferred_schemes_full_output.csv")

print(f"Total rows in dataset: {len(df)}")
print("-" * 50)

# Get value counts of the inferred schemes
distribution = df['inferred_scheme_output'].value_counts()

print("Inferred Scheme Distribution:")
print(distribution)

print("-" * 50)
print(f"Total unique schemes predicted: {distribution.nunique() if hasattr(distribution, 'nunique') else len(distribution)}")
