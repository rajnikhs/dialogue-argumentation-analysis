import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

print("Loading dataset...")
df = pd.read_csv('inferred_schemes_full_output.csv')

text_col = 'displayed_text'

# Let's use the normalization mapping function from normalize_schemes.py logic
# Or load scheme_name_only and standardize it
scheme_col = 'scheme_name_only' if 'scheme_name_only' in df.columns else 'Argumentation Scheme'
df = df.dropna(subset=[text_col, scheme_col])

def clean_scheme_name(name):
    if not isinstance(name, str):
        return "unknown"
    return name.strip().lower()

df['normalized_scheme'] = df[scheme_col].apply(clean_scheme_name)

# Also bring in topic info if posts_with_topics.csv exists, otherwise generate basic text features
try:
    topics_df = pd.read_csv('posts_with_topics.csv')
    if 'topic_name' in topics_df.columns:
        df = df.merge(topics_df[['instance_id', 'topic_name']], on='instance_id', how='left')
    else:
        df['topic_name'] = 'General'
except Exception:
    df['topic_name'] = 'General'

df['feature_text'] = df[text_col].fillna('') + " [TOPIC] " + df['topic_name'].fillna('General')

X = df['feature_text']
y = df['normalized_scheme']

# Filter out classes with fewer than 2 samples so stratified splitting doesn't fail
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_filtered = df[df['normalized_scheme'].isin(valid_classes)]

X = df_filtered['feature_text']
y = df_filtered['normalized_scheme']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

print("Vectorizing text and training classifier...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classifier = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(vectorizer, 'scheme_vectorizer.pkl')
joblib.dump(classifier, 'scheme_predictor_model.pkl')
print("\nSuccessfully retrained and saved model with normalized categories!")
