# Sentiment Analysis with NLP Models

This project applies Natural Language Processing (NLP) techniques to classify IMDB movie reviews as positive or negative.

It compares three different modeling approaches:

* Naive Bayes with Bag-of-Words / TF-IDF
* Deep Learning with LSTM + GloVe embeddings
* Transformer-based models (ALBERT, DistilBERT)

## **Dataset**

**Source:** [IMDB 50K Movie Reviews Dataset](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification/data?select=test.csv) <br />

**Size:** 50,000 labeled reviews (balanced: 25k positive / 25k negative) <br />

**Target:** Sentiment (binary classification)

## **Methodology**

**1. Exploratory Data Analysis (EDA)**

* Review length distribution (words & characters)
* Stopword analysis
* Word clouds for positive/negative reviews
* Patterns: HTML tags, emojis, excessive punctuation, slang

**2. Preprocessing**

* HTML/URL removal
* Lowercasing & contraction expansion
* Stopword removal & lemmatization
* Tokenization (NLTK & HuggingFace)

**3. Feature Engineering**

* CountVectorizer & TF-IDF for ML baseline
* Word embeddings (GloVe 100D) for LSTM
* Tokenizer + Padding for sequence models

**4. Models Trained**

* **Naive Bayes:** Baseline with DTM & TF-IDF
* **LSTM (Bi-LSTM + GlobalMaxPool):** trained with GloVe embeddings
* **Transformers:** Fine-tuned ALBERT & DistilBERT (HuggingFace Trainer)

**5. Evaluation**

* Metrics: Accuracy, Precision, Recall, F1
* Confusion matrices plotted for each model

## **Results**

* **Naive Bayes:** Good baseline, limited accuracy with %86 macro average F1-score
* **LSTM + GloVe:** Improved performance, better contextual capture with %87 macro average F1-score
* **DistilBERT / ALBERT:** Best overall performance with %94 macro average F1-score
