import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

print("Loading dataset...")
df = pd.read_csv('inferred_schemes_full_output.csv')

# Use the correct text column found in your dataset
text_col = 'displayed_text'
scheme_col = 'scheme_name_only' if 'scheme_name_only' in df.columns else 'Argumentation Scheme'

df = df.dropna(subset=[text_col, scheme_col])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs and special characters, keeping natural language
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

print("Cleaning text and extracting topics from displayed_text...")
cleaned_texts = df[text_col].apply(clean_text)

# Use robust TF-IDF parameters
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=3, max_df=0.85)
X = vectorizer.fit_transform(cleaned_texts)

n_topics = 6
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(X)

df['dominant_topic'] = W.argmax(axis=1)

feature_names = vectorizer.get_feature_names_out()
topic_labels = {}
for topic_idx, topic in enumerate(nmf.components_):
    top_features_ind = topic.argsort()[:-5:-1]
    top_words = " / ".join([feature_names[i] for i in top_features_ind])
    topic_labels[topic_idx] = f"Theme {topic_idx+1}: {top_words}"

df['topic_name'] = df['dominant_topic'].map(topic_labels)

print("Cross-tabulating cleaner topics with argumentation schemes...")
contingency_table = pd.crosstab(df['topic_name'], df[scheme_col], normalize='index')

print("\n--- Cleaned Topic vs Scheme Probabilities ---")
print(contingency_table.iloc[:, :5])

contingency_table.to_csv('topic_scheme_distribution.csv')
df.to_csv('posts_with_topics.csv', index=False)
print("\nSaved clean mapping to topic_scheme_distribution.csv and posts_with_topics.csv!")
