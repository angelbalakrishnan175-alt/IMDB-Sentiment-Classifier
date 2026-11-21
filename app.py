from flask import Flask, render_template, request
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
negation_words = {"not", "no", "never", "nor", "neither", "cannot"}

stop_words = set(stopwords.words("english")) - negation_words

lemmatizer = WordNetLemmatizer()
def clean_review(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()

    words = word_tokenize(text)

    words = [
        word for word in words
        if word.isalpha() and word not in stop_words
    ]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    review = ""

    if request.method == "POST":
        review = request.form["review"]

        cleaned_review = clean_review(review)
        review_vector = vectorizer.transform([cleaned_review])

        result = model.predict(review_vector)[0]

        probabilities = model.predict_proba(review_vector)[0]
        confidence = round(max(probabilities) * 100, 2)

        prediction = result.capitalize()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)