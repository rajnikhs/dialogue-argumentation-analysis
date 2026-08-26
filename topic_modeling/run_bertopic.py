import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

print("--- Starting BERTopic Modeling Pipeline ---")

# 1. Load the cleaned dataset
# Adjust the path if your CSV is in the root vs topic_modeling folder
input_file = "../inferred_schemes_full_output_cleaned.csv"
if not os.path.exists(input_file):
    input_file = "inferred_schemes_full_output_cleaned.csv"

df = pd.read_csv(input_file)
print(f"Loaded dataset with {len(df)} rows.")

# Extract the text column (using 'displayed_text')
docs = df['displayed_text'].dropna().astype(str).tolist()

# 2. Initialize Sentence-Transformers for embeddings
print("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Initialize and fit BERTopic
print("Training BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)

# 4. Extract topic info and save results
topic_info = topic_model.get_topic_info()
print("\nTop Topics Discovered:")
print(topic_info.head(10))

# Save topic info summary table
topic_info.to_csv("bertopic_topic_summary.csv", index=False)

# Add topic assignments back to the main DataFrame (handling missing/dropped values index alignment)
df['bertopic_id'] = -1
valid_indices = df['displayed_text'].dropna().index
df.loc[valid_indices, 'bertopic_id'] = topics

# Add human-readable topic labels representation
topic_labels = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
df['bertopic_name'] = df['bertopic_id'].map(topic_labels)

# Save final enriched dataset
output_file = "posts_with_bertopic_results.csv"
df.to_csv(output_file, index=False)
print(f"\n--- BERTopic Pipeline Complete! Saved summary to bertopic_topic_summary.csv and results to {output_file} ---")
