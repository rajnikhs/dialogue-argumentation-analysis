import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the output data
df = pd.read_csv("inferred_schemes_full_output.csv")

# Filter out script fallbacks to plot pure model predictions
valid_df = df[~df["inferred_scheme_output"].isin(["N/A", "Error"])]
counts = valid_df["inferred_scheme_output"].value_counts()

# Set up the plot style
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x=counts.values, y=counts.index, palette="Blues_r")

# Add labels and title
plt.title("Distribution of Inferred Argumentation Schemes", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Frequency (Number of Times Used)", fontsize=12)
plt.ylabel("Argumentation Scheme", fontsize=12)

plt.tight_layout()

# Save the figure
output_image = "outputs/scheme_distribution.png"
plt.savefig(output_image, dpi=300)
print(f"Figure successfully saved as {output_image}")
