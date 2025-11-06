# Clothing-Review-Sentiment-Analysis-and-Recommendation-System
An NLP-based Machine Learning Model for Classifying and Understanding Clothing Reviews 
**Overview**
This project implements an end-to-end Natural Language Processing (NLP) pipeline for analysing online clothing reviews.
It performs sentiment classification on user-written product reviews using both traditional Bag-of-Words (BoW) and embedding-based (FastText) text representations.
The system evaluates multiple vectorisation techniques to understand which representations produce more accurate predictions in a binary classification setting (Recommended = 1, Not Recommended = 0).

**Objectives**

Transform raw review text into clean, tokenised, and noise-free corpora.

Build feature representations using:

Count Vectors (Bag-of-Words)

TF-IDF weighted word embeddings (FastText)

Train and evaluate Logistic Regression models to classify sentiments.

Compare model performance between different feature representations.

**🧩 Dataset**

The dataset consists of approximately 19,600 clothing reviews, each containing fields such as:

Title

Review Text

Recommended IND (Target: 1 = recommended, 0 = not recommended)

**⚙️ Workflow**

1️⃣ Text Preprocessing

Performed tokenisation, normalisation, and cleaning using NLTK:

Removed stopwords and single-character tokens.

Pruned rare terms (TF = 1) and top-20 most frequent tokens.

Saved cleaned reviews as processed.csv and vocabulary mapping as vocab.txt.


**2️⃣ Feature Engineering**

Constructed three document representations:

Bag-of-Words (Count Vectors)

TF-IDF Features

FastText Word Embeddings (unweighted and TF-IDF weighted averages)



**3️⃣ Model Training & Evaluation**

Used Logistic Regression with 5-fold Stratified Cross-Validation to compare BoW vs. Embedding performance.

Evaluation Metrics:

Accuracy

Precision (Macro)

Recall (Macro)

F1 Score (Macro)


**Insights**

FastText TF-IDF embeddings outperformed BoW by effectively capturing semantic meaning and contextual relationships between words.

Logistic Regression provided a strong baseline due to its interpretability and robustness with high-dimensional feature spaces.

Preprocessing and vocabulary pruning significantly improved generalisation and reduced overfitting.

**🛠️ Technologies Used**

Python 3.10+

Libraries: pandas, numpy, scikit-learn, gensim, nltk, tqdm

Models: Logistic Regression

Embeddings: FastText pretrained vectors (wiki-news-300d-1M-subword.vec)

**🗂️ Project Structure**
Clothing-Review-Sentiment-Analysis-and-Recommendation-System/
│
├── task1.ipynb            # Text preprocessing pipeline
├── task2_3.ipynb          # Feature construction & classification
├── processed.csv           # Cleaned review text
├── vocab.txt               # Word:index vocabulary mapping
├── count_vectors.txt       # Bag-of-Words features
├── fasttext_unweighted.csv # Embedding-based feature matrix
├── fasttext_tfidf.csv      # TF-IDF weighted embeddings
├── requirements.txt
└── README.md

**Key Learning Outcomes**

End-to-end NLP workflow design: preprocessing → vectorisation → classification → evaluation.

Understanding trade-offs between symbolic (BoW) and semantic (embeddings) text representations.

Practical application of FastText embeddings and TF-IDF weighting for sentiment analysis.

Data-driven performance comparison of traditional ML vs. embedding-based NLP techniques.

**👩‍💻 Author**

Gayathri Devi Thotappa
Master of Data Science, RMIT University
https://github.com/gayathri19-tech
