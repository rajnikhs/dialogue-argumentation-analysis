import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading dataset and models...")
df = pd.read_csv('inferred_schemes_full_output.csv')
text_col = 'displayed_text'
scheme_col = 'scheme_name_only' if 'scheme_name_only' in df.columns else 'Argumentation Scheme'

# Normalize scheme names in dataset
df['normalized_scheme'] = df[scheme_col].astype(str).str.strip().str.lower()
df = df.dropna(subset=[text_col, 'normalized_scheme'])

# Load trained vectorizer and model
try:
    vectorizer = joblib.load('scheme_vectorizer.pkl')
    classifier = joblib.load('scheme_predictor_model.pkl')
except Exception as e:
    print(f"Error loading model files: {e}. Please run train_scheme_predictor.py first.")
    exit(1)

# Fit a separate document vectorizer for semantic similarity search over text
print("Building semantic text search index...")
text_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
doc_term_matrix = text_vectorizer.fit_transform(df[text_col].fillna(''))

def find_similar_arguments(query_text, top_k=3):
    print(f"\nQuery Post: \"{query_text[:120]}...\"")
    
    # 1. Predict the scheme for the query text (using a dummy topic context or general)
    query_feature = query_text + " [TOPIC] General"
    query_vec = vectorizer.transform([query_feature])
    predicted_scheme = classifier.predict(query_vec)[0]
    print(f"-> Predicted Argumentation Scheme: **{predicted_scheme}**")
    
    # 2. Filter historical database for posts matching the predicted scheme
    subset_df = df[df['normalized_scheme'] == predicted_scheme].reset_index(drop=True)
    
    if len(subset_df) == 0:
        print(f"No historical arguments found matching scheme: {predicted_scheme}. Falling back to entire dataset.")
        subset_df = df.copy()
        subset_matrix = doc_term_matrix
    else:
        # Get matrix indices for the filtered subset
        subset_indices = df[df['normalized_scheme'] == predicted_scheme].index
        subset_matrix = doc_term_matrix[subset_indices]
        
    # 3. Compute cosine similarity between the query and historical texts
    query_doc_vec = text_vectorizer.transform([query_text])
    similarities = cosine_similarity(query_doc_vec, subset_matrix).flatten()
    
    # Get top-k matches
    top_indices = similarities.argsort()[::-1][:top_k]
    
    print(f"\n--- Top {top_k} Similar Historical Arguments (Scheme: {predicted_scheme}) ---")
    for i, idx in enumerate(top_indices):
        match_row = subset_df.iloc[idx] if len(subset_df) == len(df) else subset_df.iloc[idx]
        score = similarities[idx]
        print(f"\n[{i+1}] Similarity Score: {score:.4f}")
        print(f"User: {match_row.get('user', 'Unknown')}")
        print(f"Text snippet: {str(match_row[text_col])[:300]}...")

# Test with a sample query from your dataset
sample_query = df[text_col].iloc[0] if len(df) > 0 else "Test dilemma text"
find_similar_arguments(sample_query, top_k=3)

