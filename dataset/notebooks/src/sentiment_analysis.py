# Import required libraries

import pandas as pd
import numpy as np
import nltk
import time
import re

from pathlib import Path
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# Define project paths

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_DIR / "IMDB Dataset.csv"

# Load the IMDB dataset

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five reviews:")
print(df.head())
# 1. Dataset Exploration

print("\n--- DATASET EXPLORATION ---")

print("Total reviews:", len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment percentages:")
print(df["sentiment"].value_counts(normalize=True) * 100)

# Calculate majority baseline accuracy
baseline = df["sentiment"].value_counts(normalize=True).max() * 100

print("\nMajority baseline accuracy:", baseline)

# Calculate review word counts
df["word_count"] = df["review"].apply(lambda x: len(x.split()))

print("\nWord count statistics:")
print("Mean:", df["word_count"].mean())
print("Median:", df["word_count"].median())
print("Mode:", df["word_count"].mode().tolist())
print("Variance:", df["word_count"].var())
print("Standard deviation:", df["word_count"].std())
print("Skewness:", df["word_count"].skew())

print("\nMean word count by sentiment:")
print(df.groupby("sentiment")["word_count"].mean())

print("\nDuplicate rows:", df.duplicated().sum())
# 2. Text Preprocessing

# Initialize stopwords and lemmatizer once
stop_words = set(stopwords.words("english"))

# Preserve important negation words
negation_words = {"not", "no", "never", "nor", "neither", "cannot"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_review(text):
    """
    Clean a movie review by removing HTML, stopwords
    and non-alphabetic tokens, then apply lemmatization.
    """

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Convert to lowercase
    text = text.lower()

    # Tokenize the review
    words = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    words = [
        word for word in words
        if word.isalpha() and word not in stop_words
    ]

    # Apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# Measure cleaning time for 1,000 reviews
start_time = time.time()

for review in df["review"].head(1000):
    clean_review(review)

end_time = time.time()

print("\n--- TEXT PREPROCESSING ---")
print("Time for 1,000 reviews:", round(end_time - start_time, 2), "seconds")

# Clean the complete dataset
df["cleaned_review"] = df["review"].apply(clean_review)

# Count empty cleaned reviews
print("Empty cleaned reviews:", (df["cleaned_review"] == "").sum())
# 3. Split the Dataset

X = df["cleaned_review"]
y = df["sentiment"]

# Stratified split maintains the sentiment balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- DATA SPLITTING ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. TF-IDF Feature Extraction

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=50000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\n--- TF-IDF ---")
print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)
# 5. Train the Sentiment Classification Model

model = MultinomialNB(alpha=0.5)

# Train the model using TF-IDF features
model.fit(X_train_tfidf, y_train)

# Predict sentiment for test reviews
y_pred = model.predict(X_test_tfidf)

print("\n--- MODEL TRAINING ---")
print("Model training completed successfully!")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("Baseline Accuracy:", round(baseline, 2), "%")

# Classification report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))
# 6. Identify Important Words for Each Sentiment

feature_names = tfidf.get_feature_names_out()

print("\n--- TOP 20 WORDS BY SENTIMENT ---")

for sentiment_index, sentiment in enumerate(model.classes_):

    top_indices = model.feature_log_prob_[sentiment_index].argsort()[-20:][::-1]

    print("\nTop 20 terms for", sentiment.upper(), "reviews:")
    print(feature_names[top_indices])
    # 7. Bigram Analysis

bigram_vectorizer = TfidfVectorizer(
    ngram_range=(2, 2),
    min_df=5,
    max_features=50000
)

X_bigram = bigram_vectorizer.fit_transform(X_train)

bigram_model = MultinomialNB(alpha=0.5)
bigram_model.fit(X_bigram, y_train)

bigram_names = bigram_vectorizer.get_feature_names_out()

print("\n--- TOP BIGRAMS ---")

for sentiment_index, sentiment in enumerate(bigram_model.classes_):

    indices = bigram_model.feature_log_prob_[sentiment_index].argsort()[-10:][::-1]

    print("\nTop bigrams for", sentiment.upper())
    print(bigram_names[indices])