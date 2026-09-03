import pandas as pd

# Load your full output file
df = pd.read_csv("qt_inferred_schemes_full_output.csv")

# These are the exact anchor texts provided inside your few-shot prompt definitions
anchor_examples = {
    "Direct Ad Hominem": "combat operations in Afghanistan finished in 2014. Mehdi Hasan is living in the past. Mehdi Hasan is massively misrepresenting the Afghanistan situation.",
    "Inconsistent Commitment": "politicians knew. that is the worst part about it, to see politicians this week saying, \"I'm shocked at this collapse, shocked it didn't work out the way ... \".",
    "Cause To Effect": "businesses are closing all over the place. the hospitality industry is literally on its knees.",
    "Established Rule": "we should have learned from what happened in Vietnam. we haven't done well but we haven't done as badly as some other countries.",
    "Verbal Classification": "women get far more abuse than men in social media and black people. Naomi Osaka is a black woman and she gets tremendous abuse, no doubt on social media.",
    "Analogy": "the Afghanistan war is on the same level of a failure as the Iraq war. what we've done in Afghanistan seriously deserves a huge accounting.",
    "Example": "teachers have worked so hard over the last few months. the teachers of Sam Unkown's boys have been such a credit.",
    "Best Explanation": "there have been a series of political and strategic failures over a very long period of time that led to here. the current situation has long and deep roots.",
    "Ignorance": "levelling up is not about investing in our people. we don't know what levelling up is for."
}

print("--- Testing Model Behavior on Prompt's Own Anchor Examples ---")
print(f"{'Expected Scheme':<25} | {'Model Prediction':<25} | {'Match?'}")
print("-" * 75)

matches = 0
found_count = 0

for expected_scheme, snippet in anchor_examples.items():
    # Find rows containing this snippet
    match_row = df[df['text'].str.contains(snippet[:30], case=False, na=False)]
    if not match_row.empty:
        found_count += 1
        pred = match_row.iloc[0]['inferred_scheme_output']
        is_match = expected_scheme.lower() in str(pred).lower()
        if is_match:
            matches += 1
        print(f"{expected_scheme:<25} | {str(pred):<25} | {'YES' if is_match else 'NO'}")

print("-" * 75)
print(f"Matched {matches} out of {found_count} anchor examples found in dataset.")
