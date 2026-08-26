import pandas as pd
import re

# 1. Load the dataset
input_file = "inferred_schemes_full_output.csv"
df = pd.read_csv(input_file)

# 2. Function to extract just the clean scheme name from the LLM output
def extract_scheme_name(text):
    if pd.isna(text) or not isinstance(text, str):
        return "N/A"
    
    # Remove markdown bold asterisks
    cleaned = text.replace("*", "").strip()
    
    # Try to match patterns like "scheme name: Popular Practice" or "SCHEME: ..."
    match = re.search(r'(?:scheme\s*name[:\s]*|scheme[:\s]*)([^\n\r]+)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: Take the first line if it's short and clean
    first_line = cleaned.split('\n')[0].strip()
    # Remove leading labels if present
    first_line = re.sub(r'^(scheme name|scheme)[:\s]*', '', first_line, flags=re.IGNORECASE).strip()
    
    return first_line if len(first_line) < 45 else "Unknown"

# 3. Apply the extraction and put it as the last column on the right
df['clean_scheme_name'] = df['inferred_scheme_output'].apply(extract_scheme_name)

# Reorder so 'clean_scheme_name' is absolute last (furthest right column)
cols = [col for col in df.columns if col != 'clean_scheme_name'] + ['clean_scheme_name']
df = df[cols]

# 4. Save the updated file
output_file = "inferred_schemes_full_output_cleaned.csv"
df.to_csv(output_file, index=False)
print(f"Successfully added 'clean_scheme_name' as the rightmost column and saved to {output_file}!")
