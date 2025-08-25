import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
from normalizer import TextNormalizer
import os

# Load and prepare the dataset
# Note: You’ll need to download this from Kaggle beforehand
tweets_df = pd.read_csv("data/Tweets.csv")

# Only keep what we actually care about (might regret dropping later…)
tweets_df = tweets_df[['text', 'airline_sentiment']]

# Remove rows with missing stuff
tweets_df = tweets_df.dropna()

# Just to make life easier, map sentiments to numbers
# (Order depends on the dataset's current ordering — could change if reloaded)
sentiment_to_num = {sent: idx for idx, sent in
                    enumerate(tweets_df['airline_sentiment'].unique())}
tweets_df['label'] = tweets_df['airline_sentiment'].map(sentiment_to_num)

# Features and labels
texts = tweets_df['text']
labels = tweets_df['label']

# Train/test split — 80/20 because… tradition.
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,  # For reproducibility
    stratify=labels   # Keeps class balance intact
)

# Make sure slang file path works (case-insensitive fix for Windows/Linux)
slang_path = os.path.join("resources", "slang.json")
if not os.path.exists(slang_path):
    slang_path = "Slang.json"  # fallback if resources folder not used

# Build the pipeline
tweet_pipeline = Pipeline([
    ('normalize', TextNormalizer(slang_path=slang_path)),  # custom text cleaner
    ('vectorize', TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),  # 1-grams + 2-grams
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'  # handles imbalance
    ))
])

# Train the model
tweet_pipeline.fit(X_train, y_train)

# Make predictions
preds = tweet_pipeline.predict(X_test)

# Evaluate
print("Model Accuracy:", accuracy_score(y_test, preds))
print("\nDetailed classification report:")
print(classification_report(y_test, preds, target_names=sentiment_to_num.keys()))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Save the pipeline
os.makedirs("models", exist_ok=True)  # ensure models folder exists
dump(tweet_pipeline, "models/sentiment_pipeline.joblib")
print("Pipeline saved to 'models/sentiment_pipeline.joblib' — done!")

# Sample inputs you can change to test code
sample_texts = [
    "I really enjoyed my flight today!",
    "The delay was unacceptable and staff were rude.",
    "Flight was okay, nothing special.",
    "Great service and friendly crew!",
    "This was the worst airline experience ever."
]

print("\n=== Example Predictions with Probabilities ===")
for text in sample_texts:
    # Get prediction probabilities
    probabilities = tweet_pipeline.predict_proba([text])[0]
    prediction = tweet_pipeline.predict([text])[0]
    
    # Map the numerical prediction back to the sentiment label
    sentiment_label = list(sentiment_to_num.keys())[list(sentiment_to_num.values()).index(prediction)]
    
    print(f"'{text}'")
    print(f"→ Prediction: {sentiment_label}")
    for i, (sentiment, idx) in enumerate(sentiment_to_num.items()):
        print(f"  {sentiment}: {probabilities[idx]:.3f}")
    print()
