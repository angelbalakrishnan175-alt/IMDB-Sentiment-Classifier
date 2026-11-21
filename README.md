\# IMDB Movie Review Sentiment Analysis



\## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify IMDB movie reviews as positive or negative.



The model is trained on 50,000 movie reviews and uses TF-IDF feature extraction with a Multinomial Naive Bayes classifier.



\## Dataset

\- Total reviews: 50,000

\- Positive reviews: 25,000

\- Negative reviews: 25,000

\- Dataset: IMDB Movie Reviews



\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- NLTK

\- Scikit-learn

\- BeautifulSoup

\- TF-IDF Vectorization

\- Multinomial Naive Bayes



\## Project Workflow

1\. Dataset loading and exploration

2\. Statistical analysis and visualization

3\. Text preprocessing

4\. HTML tag removal

5\. Tokenization

6\. Stopword removal while preserving negations

7\. Lemmatization

8\. TF-IDF feature extraction

9\. Model training

10\. Model evaluation



\## Model Performance



| Metric | Result |

|---|---:|

| Model Accuracy | 88.63% |

| Baseline Accuracy | 50% |

| Training Samples | 40,000 |

| Testing Samples | 10,000 |



\## Confusion Matrix



```text

\[\[4374  626]

&#x20;\[ 539 4461]]

```



\## Key Findings

\- The dataset contains 50,000 balanced reviews.

\- The model achieved 88.63% accuracy.

\- Text preprocessing removed HTML tags and unnecessary words.

\- Negation words were preserved to retain sentiment meaning.

\- TF-IDF helped convert text into numerical features.



\## Limitations

\- Sarcasm and irony can be difficult to identify.

\- Word meanings may depend on context.

\- The model assumes conditional independence between features.

\- Performance may vary on text outside movie reviews.



\## Project Structure



```text

IMDB\_Sentiment\_Classifier/

│

├── dataset/

│   └── notebooks/

│       └── src/

│           └── sentiment\_analysis.py

│

├── outputs/

│   └── results/

│       ├── project\_results.txt

│       └── console\_output.txt

│

├── IMDB Dataset.csv

└── README.md

```



\## Conclusion

This project demonstrates how Natural Language Processing and machine learning can be used to analyze movie reviews and predict their sentiment.



The Multinomial Naive Bayes classifier achieved 88.63% accuracy, performing significantly better than the 50% majority baseline.ss

