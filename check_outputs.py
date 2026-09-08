import pandas as pd

# Load your output file
df = pd.read_csv("inferred_schemes_full_output.csv")

print("Columns in file:", df.columns.tolist())
print("\nFirst 10 rows of text, stance, and model prediction:")

# Loop through the first 10 rows safely using the correct columns
for idx in range(min(10, len(df))):
    text_sample = str(df.loc[idx, 'displayed_text'])[:50] + "..."
    stance = df.loc[idx, 'stance']
    prediction = df.loc[idx, 'inferred_scheme_output']
    print(f"[{idx}] Text: {text_sample} | Stance: {stance} | Pred: {prediction}")
