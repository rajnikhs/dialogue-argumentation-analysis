import glob
import pandas as pd

all_schemes = set()
for file in glob.glob('qt_schemes_repo/QT-Schemes/*.csv'):
    df = pd.read_csv(file, sep=';')
    all_schemes.update(df['scheme_gold'].dropna().unique())

print("All unique schemes in QT-Schemes corpus:")
for s in sorted(all_schemes):
    print(f"  - '{s}'")
print(f"\nTotal unique schemes: {len(all_schemes)}")
