import matplotlib.pyplot as plt
import pandas as pd

# Load your normalized summary data
df = pd.read_csv('normalized_scheme_summary.csv')

# Print columns to verify names if needed
print('Columns found:', df.columns.tolist())

# Sort using the correct capitalized column name ('Count')
df_sorted = df.sort_values(by='Count', ascending=True)

# Generate the bar chart
plt.figure(figsize=(10, 8))
plt.barh(
    df_sorted['Argumentation Scheme'], df_sorted['Count'], color='steelblue'
)
plt.xlabel('Frequency / Count', fontsize=12)
plt.ylabel('Argumentation Scheme', fontsize=12)
plt.title(
    'Distribution of Argumentation Schemes (Excluding Verdicts)',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()

# Save the chart
plt.savefig('scheme_distribution.png', dpi=300)
plt.close()

print('Successfully generated scheme_distribution.png!')
