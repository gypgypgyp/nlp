# nlp

# Shakespeare Autocomplete

This project implements an autocomplete system based on the works of William Shakespeare. It processes Shakespeare’s text, builds a frequency-based Trie data structure, and suggests the 10 most common words starting with a given prefix.

## Features
Processes Shakespeare's text: Cleans punctuation, converts to lowercase, and extracts unique words ordered by frequency.

Efficient search using a Trie: Allows fast prefix-based autocomplete suggestions.
Two Interfaces:

CLI Version (assignment1_cli.py): Run in the terminal for quick testing.
Web App (assignment1.py): Uses Streamlit for an interactive GUI.


# TweetSentimentNN

Neural Network-Based Sentiment Analysis for Tweets

This project implements a **two-layer neural network** from scratch to predict the **sentiment of tweets** using **NumPy only**.

## Features
- **Neural Network Architecture**:
  - Input layer: Extracted tweet features
  - Hidden layer: Sigmoid activation
  - Output layer: Binary classification (Positive/Negative)
- **Binary Cross-Entropy (BCE) Loss**
- **Gradient Descent Optimization**
- **Training & Testing Loss Curve Visualization**
- **Sentiment Prediction for Sample Tweets**



# AutoCorrectDP

This project implements a **basic autocorrect system** using **probability-based word correction** and **dynamic programming** for computing **minimum edit distance**.

## Features
- **Process a Text Corpus**: Reads a corpus, normalizes words, and computes word probabilities.
- **Identify Probable Words**: Generates candidate corrections based on:
  - Deletion
  - Swapping adjacent letters
  - Replacing a letter
  - Inserting a letter
- **Compute Minimum Edit Distance**: Implements **dynamic programming** to calculate the **minimum number of operations** required to convert one word into another.



# TweetNGramPredictor

This project implements an **N-Gram-based auto-correct and next-word prediction system** using **Twitter data**. It builds **bigram** and **trigram models**, estimates word probabilities, and predicts the most probable next word.

## Features
- **Preprocessing & Tokenization**:
  - Reads and tokenizes Twitter data.
  - Handles **Out-of-Vocabulary (OOV)** words using `<unk>`.
  - Splits data into **train/test** sets.
- **N-Gram Counting**:
  - Computes **unigram, bigram, and trigram** counts.
  - Uses **Laplace smoothing (k=1)** for better probability estimation.
- **Next-Word Prediction**:
  - Predicts the most likely next word for a given phrase.
  - Uses an **N-Gram probability model**.
- **Style-Based Prediction (Extra Credit)**:
  - Trains **Hemingway, Shakespeare, and Dickinson**-style N-Gram models.
  - Predicts the **most stylistically probable next word**.



# SVD_W2V

This project implements word embedding techniques using a dataset of arXiv paper titles. It explores:

Singular Value Decomposition (SVD) - A factorization-based method.
Word2Vec (Skip-gram with Negative Sampling) - A neural-network-based approach.
The dataset consists of scientific paper titles from arXiv, where word embeddings are learned based on word co-occurrence.
