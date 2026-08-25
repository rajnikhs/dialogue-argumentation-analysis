import pandas as pd
import re

# Load your output CSV file
input_file = 'inferred_schemes_full_output.csv'
df = pd.read_csv(input_file)

def extract_scheme_name(text):
    if pd.isna(text):
        return ""
    text_str = str(text)
    # Search for text enclosed in double asterisks, e.g., **Direct Ad Hominem**
    match = re.search(r'\*\*(.*?)\*\*', text_str)
    if match:
        return match.group(1).strip()
    return text_str.strip()

# Create the new column
df['scheme_name_only'] = df['inferred_scheme_output'].apply(extract_scheme_name)

# Save the updated dataset back to CSV and Excel
df.to_csv('inferred_schemes_full_output.csv', index=False)
df.to_excel('inferred_schemes_full_output.xlsx', index=False)

print("Successfully added the 'scheme_name_only' column and saved files!")
