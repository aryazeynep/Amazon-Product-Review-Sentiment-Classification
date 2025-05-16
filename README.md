# Amazon-Product-Review-Sentiment-Classification
## 📌 About the Project

The aim of this project is to classify Amazon product reviews as positive or negative using text-based sentiment analysis. We implemented a Naive Bayes classifier from scratch using the Bag of Words (BoW) approach with unigram and bigram features. The dataset includes 72,500 balanced product reviews (1 to 5 stars), and we apply data preprocessing, feature extraction, model training, and evaluation.

## 🛠️ Technologies Used
Python 3

Custom implementation of Naive Bayes

Bag of Words (BoW)

Unigrams and Bigrams

TF-IDF (for optional comparison)

Word Embedding + Logistic Regression (Bonus)

Libraries (only for bonus part):

gensim (for Word2Vec)

nltk (for additional NLP tools)

scikit-learn (for Logistic Regression and evaluation metrics)

## 📂 Dataset Information
Total reviews: 72,500

Balanced across 5 classes (1 to 5 stars)

Each review includes:

- Title

- Content

- Star rating (1–5)

Label Mapping:

1 & 2 stars → Negative

4 & 5 stars → Positive

3 stars → Discarded (with explanation)

We apply weighted scoring for sentiment intensity:
1-star is more negative than 2-star, 5-star is more positive than 4-star.

## ✅ Project Steps
**Data Preprocessing**

Clean text (punctuation, stopwords, lowercase)

Merge title and content with adjustable weights

**Feature Extraction**

Implemented Bag of Words model with unigrams and bigrams

Applied custom dictionary construction

Handled unknown words using Laplace Smoothing

**Model Implementation**

Custom Naive Bayes classifier (from scratch)

Logarithmic probabilities to prevent underflow

**Evaluation Metrics**

Accuracy

Precision

Recall

F1-Score

**Bonus Implementation**

Word Embeddings (Word2Vec, GloVe)

Logistic Regression using pre-trained embeddings

Comparison with baseline custom model and NLTK-based approach

## 📊 Results
Evaluation was conducted using an 80-20 train-test split.
Each configuration (unigram, bigram, word embeddings) was compared based on performance metrics.

Performance metrics such as Accuracy, Precision, Recall, and F1-Score were computed for each model and setting.

