import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

print("=== STEP 1: Processing Topics & Saving Output ===")
df = pd.read_csv('inferred_schemes_full_output.csv')
text_col = 'displayed_text'
scheme_col = 'scheme_name_only' if 'scheme_name_only' in df.columns else 'Argumentation Scheme'

df = df.dropna(subset=[text_col, scheme_col])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

cleaned_texts = df[text_col].apply(clean_text)
vectorizer_nmf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=3, max_df=0.85)
X_nmf = vectorizer_nmf.fit_transform(cleaned_texts)

nmf = NMF(n_components=6, random_state=42)
W = nmf.fit_transform(X_nmf)
df['dominant_topic'] = W.argmax(axis=1)

feature_names = vectorizer_nmf.get_feature_names_out()
topic_labels = {}
for topic_idx, topic in enumerate(nmf.components_):
    top_features_ind = topic.argsort()[:-5:-1]
    top_words = " / ".join([feature_names[i] for i in top_features_ind])
    topic_labels[topic_idx] = f"Theme {topic_idx+1}: {top_words}"

df['topic_name'] = df['dominant_topic'].map(topic_labels)

# Save Topic Summary Output File
with open('topic_grouping_output.txt', 'w') as f:
    f.write("=== AITA POSTS TOPIC GROUPING SUMMARY ===\n\n")
    for idx, label in topic_labels.items():
        count = len(df[df['dominant_topic'] == idx])
        f.write(f"{label} (Assigned Posts: {count})\n")
print("Saved topic_grouping_output.txt")


print("\n=== STEP 2: Training Classifier & Saving Evaluation Report ===")
def clean_scheme_name(name):
    if not isinstance(name, str):
        return "unknown"
    return name.strip().lower()

df['normalized_scheme'] = df[scheme_col].apply(clean_scheme_name)
df['feature_text'] = df[text_col].fillna('') + " [TOPIC] " + df['topic_name'].fillna('General')

class_counts = df['normalized_scheme'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_filtered = df[df['normalized_scheme'].isin(valid_classes)]

X = df_filtered['feature_text']
y = df_filtered['normalized_scheme']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_vec = clf_vectorizer.fit_transform(X_train)
X_test_vec = clf_vectorizer.transform(X_test)

classifier = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# Save Classification Evaluation Output File
with open('classification_evaluation_report.txt', 'w') as f:
    f.write("=== SCHEME PREDICTOR CLASSIFICATION REPORT ===\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write("Classification Metrics:\n")
    f.write(report)
print("Saved classification_evaluation_report.txt")


print("\n=== STEP 3: Running Retrieval & Saving Historical Matches Output ===")
text_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
doc_term_matrix = text_vectorizer.fit_transform(df[text_col].fillna(''))

sample_query = df[text_col].iloc[0]
query_feature = sample_query + " [TOPIC] General"
query_vec = clf_vectorizer.transform([query_feature])
predicted_scheme = classifier.predict(query_vec)[0]

subset_indices = df[df['normalized_scheme'] == predicted_scheme].index
subset_df = df[df['normalized_scheme'] == predicted_scheme].reset_index(drop=True)
subset_matrix = doc_term_matrix[subset_indices]

query_doc_vec = text_vectorizer.transform([sample_query])
similarities = cosine_similarity(query_doc_vec, subset_matrix).flatten()
top_indices = similarities.argsort()[::-1][:3]

# Save Retrieved Arguments Output File
with open('retrieved_arguments_output.txt', 'w') as f:
    f.write("=== SIMILAR HISTORICAL ARGUMENTS RETRIEVAL OUTPUT ===\n\n")
    f.write(f"Query Post: \"{sample_query}\"\n")
    f.write(f"Predicted Scheme: {predicted_scheme}\n\n")
    f.write("--- Top 3 Similar Matches ---\n")
    for i, idx in enumerate(top_indices):
        match_row = subset_df.iloc[idx]
        score = similarities[idx]
        f.write(f"\n[{i+1}] Similarity Score: {score:.4f}\n")
        f.write(f"User: {match_row.get('user', 'Unknown')}\n")
        f.write(f"Text: {match_row[text_col]}\n")
print("Saved retrieved_arguments_output.txt")

print("\nAll output files successfully generated!")
