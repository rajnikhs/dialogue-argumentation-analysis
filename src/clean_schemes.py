import glob
import pandas as pd

all_schemes = set()
for file in glob.glob('qt_schemes_repo/QT-Schemes/*.csv'):
    df = pd.read_csv(file, sep=';')
    cleaned = df['scheme_gold'].dropna().str.strip().str.title()
    all_schemes.update(cleaned.unique())

print("Normalized unique schemes:")
for s in sorted(all_schemes):
    print(f"  - '{s}'")
print(f"\nTotal normalized schemes: {len(all_schemes)}")
