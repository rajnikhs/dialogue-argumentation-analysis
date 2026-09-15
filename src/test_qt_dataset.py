import pandas as pd

# Load one of the QT-Schemes files using semicolon delimiter
df = pd.read_csv('qt_schemes_repo/QT-Schemes/10June2021.csv', sep=';')

print("--- QT-Schemes Sample Data Loaded Successfully ---")
print(f"Total rows in file: {len(df)}")
print("\nColumn names:", df.columns.tolist())
print("\nGold scheme distribution:")
print(df['scheme_gold'].value_counts())

print("\nSample rows:")
print(df.head(3))
