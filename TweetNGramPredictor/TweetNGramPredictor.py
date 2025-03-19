import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('.')
import pandas as pd
import numpy as np
import re
from collections import Counter
import utils
from collections import defaultdict

class NGramModel():
    """
    This class holds your n-gram model and all its parameters, which include:

        - n_gram_counts: Dictionary mapping n-grams to their frequencies.
        - n_plus1_gram_counts: Dictionary mapping (n+1)-grams to their frequencies.
        - vocabulary: A set of unique words.
        - special_tokens: An instance of SpecialTokens.
        - k: Smoothing factor for probability estimation.
    """
    def __init__(self, n_gram_counts,
               n_plus1_gram_counts, vocabulary, special_tokens, k=1):
        self.n_gram_counts = n_gram_counts
        self.n_plus1_gram_counts = n_plus1_gram_counts
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.k = k

class SpecialTokens:
    """ 
    Class of special tokens
    """
    def __init__(self, start_token = "<s>", end_token = "<e>", unknown_token = "<unk>"):
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

# %% [markdown]
# # Question 1: Preprocessing and Vocabulary

# %%

#@title Question 1

def preprocess_data(filename, count_threshold, special_tokens,
                    sample_delimiter='\n', split_ratio=0.8):
    """
    Ungraded: You do not need to change this function.

    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" .
    Args:
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        training_data = list of lists denoting tokenized sentence. This looks like
                        the following:
 
                        [ ["this", "<unk>", "example"], 
                          ["another", "sentence", "<unk>", "right"],
                         ...
                        ] 
        test_data = Same format as above.
        vocabulary = list of vocabulary words. This looks like the following:

                        ["vocab-word-1", "vocab-word-2", etc.]
    """

    # Create sentences and tokenize the data to create a list of strings. 
    tokenized_data = read_and_tokenize_sentences(filename, sample_delimiter)

    # Create the training / test splits
    train_size = int(len(tokenized_data) * split_ratio)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    # Get the closed vocabulary using the train data. only use the words with large frequency for the vocab.
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with unknown token
    train_data_replaced = replace_oov_words_by_unk(
        train_data, vocabulary, unknown_token = special_tokens.unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(
        test_data, vocabulary, unknown_token = special_tokens.unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary

def preprocess_data_test():
    """
    Ungraded: You can use this function to test out preprocess_data. 
    """
    tmp_train = "the sky is blue.\nleaves are green.\nsmell all the roses."
    tmp_test = "roses are red."

    with open('tmp_data.txt', 'w') as f:
      f.write(str(tmp_train) + '\n')
      f.write(str(tmp_test) + '\n')

    special_tokens = SpecialTokens()
    count_threshold = 1

    tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(
        "tmp_data.txt", count_threshold, special_tokens, split_ratio = 0.75)

    assert tmp_test_repl == [['roses', 'are', '<unk>', '.']] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', 
                         special_tokens.end_token]] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', '.',
                         special_tokens.end_token]], \
      print("tmp_test_repl is not correct")

    assert tmp_train_repl == [['the', 'sky', 'is', 'blue', '.'],
                              ['leaves', 'are', 'green', '.'],
                              ['smell', 'all', 'the', 'roses', '.']] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', 
                               special_tokens.end_token]] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', '.',
                               special_tokens.end_token]], \
      print("tmp_train_repl is not correct")

    print("\033[92m Successful test")

    return 

#@title Q1.1 Read / Tokenize Data from Sentences

def read_and_tokenize_sentences(filename, sample_delimiter="\n"):
    """
    Args:
        - filename = (e.g., "en_US.twitter.txt")
        - sample_delimiter = delimits each sample (i.e., each tweet)
    Example usage: 
       $ read_and_tokenize_sentences(sentences) 
       [['sky', 'is', 'blue', '.'],
        ['leaves', 'are', 'green'],
        ['roses', 'are', 'red', '.']]A

    You can use nltk's tokenize function here.
       nltk.word_tokenize(sentence)
    """
    
    # open the file and split the sentences using delimiter (\n)
    with open(filename,'r',encoding = 'utf-8') as f:
        sentences = f.read().strip().split(sample_delimiter)
    
    # Tokenize each sentence into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    
    return tokenized_sentences

# Function to get words with frequency greater than or equal to count_threshold
def get_words_with_nplus_frequency(train_data, count_threshold):
    # initialize dictionary to store the frequency of words
    word_counts = defaultdict(int)
    for sentence in train_data:
        for word in sentence:
            word_counts[word] += 1
    # get the words with frequency lareger than or equal to count_threshold
    frequent_words = {word for word, count in word_counts.items() if count >= count_threshold}

    return frequent_words

#@title Q1.2 Replace OOV Words with Special Token

def replace_oov_words_by_unk(data, vocabulary, unknown_token="<unk>"):
    # replace the token not in the vocab with <unk>
    processed_data = [[word if word in vocabulary else unknown_token for word in sentence] for sentence in data]

    return processed_data

preprocess_data_test()

# %% [markdown]
# # Question 2: N-Gram Counting
# 

# %%
#@title Q2 Count N-Grams
def count_n_grams(data, n, special_tokens):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: Number of words in a sequence
        special_tokens: A structure that contains:
          - start_token = "<s>"
          - end_token = "<e>"
          - unknown_token = "unk"

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    # Initialize dictionary of n-grams and their counts
    n_grams = defaultdict(int)

    for sentence in data:
        # add n-1 <s> to the beginning and 1 <e> to the ending
        sentence = [special_tokens.start_token] * (n - 1) + sentence + [special_tokens.end_token]
        # calculate frequency for each n_gram
        for i in range(len(sentence) - n + 1):
            # use tuple datatype here because it can be the key for dict in python
            # tuple is ordered (different from set)
            n_gram = tuple(sentence[i:i + n])
            n_grams[n_gram] += 1
    
    return dict(n_grams)


def count_n_grams_test():
    """
    Ungraded: You can use this function to test out count_n_grams. 
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, _ = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    # print(sentences)
    # print("-----")

    received = count_n_grams(sentences, 2, SpecialTokens())
    expected = { ('<s>', 'i'): 1,
      ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2,
      ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}

    # print(received)
    # print(expected)
    assert received == expected, ("Received: \n", received, 
                                       "\n\nExpected: \n", expected)

    print("\033[92m Successful test")

    unique_words = list(set([word for sentence in sentences for word in sentence]))

    bigram_counts = count_n_grams(sentences, 2, SpecialTokens())
    print("\nBigram probabilities:")
    display(make_probability_matrix(bigram_counts, unique_words, k=1))

    return

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k=1):
    """
    Convert the count matrix into a probability matrix with smoothing.

    Args:
        n_plus1_gram_counts: Dictionary of (n+1)-gram counts
        vocabulary: List of unique words
        k: Smoothing parameter

    Returns:
        A pandas DataFrame representing the probability matrix
    """
    count_matrix = utils.make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k  # Apply smoothing
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1) + k * len(vocabulary), axis=0)

    return prob_matrix

count_n_grams_test()


# %% [markdown]
# # Question 3: Estimate the Probabilities

# %%
#@title Q3 Estimate the Probabilities

def estimate_probabilities(context_tokens, ngram_model):
    """
    Estimate the probabilities of a next word using the n-gram counts
    with k-smoothing

    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        ngram_model: a structure that contains:
            - n_gram_counts: Dictionary of counts of n-grams
            - n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            - vocabulary_size: number of words
            - k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to probability
    """
    #store the result
    probabilities = {}
    # convert context tokens to tuple as n-gram
    context_tuple = tuple(context_tokens)
    # get the n-gram frequency
    context_count = ngram_model.n_gram_counts.get(context_tuple,0)

    # print(f"Context Tuple: {context_tuple}")
    # print(f"Context Count: {context_count}")

    # Include vocabulary and special tokens
    ngram_model.vocabulary.add(ngram_model.special_tokens.unknown_token)
    ngram_model.vocabulary.add(ngram_model.special_tokens.end_token)
    
    for word in ngram_model.vocabulary:
        # Construct the (n+1)-gram by appending the current word to the context
        n_plus1_gram = context_tuple + (word,)
        # Get the count of the (n+1)-gram from the (n+1)-gram counts
        # If the (n+1)-gram is not found, default to 0
        count = ngram_model.n_plus1_gram_counts.get(n_plus1_gram, 0)
        # Compute the probability using k-smoothing (Laplace smoothing)
        # Formula: P(word | context) = (count(n+1-gram) + k) / (count(n-gram) + k * |vocabulary|)
        probabilities[word] = (count + ngram_model.k) / \
                              (context_count + ngram_model.k * len(ngram_model.vocabulary))
    # print("-------")
    # print(probabilities)
    # print("-------")
    return probabilities


def estimate_probabilities_test():
    """
    Ungraded: You can use this function to test out estimate_probabilities. 
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, vocabulary = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    # unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = count_n_grams(sentences, 1, SpecialTokens())
    bigram_counts = count_n_grams(sentences, 2, SpecialTokens())

    # print("Bigram counts:", bigram_counts.keys())
    # print("Trigram counts:", unigram_counts.keys())

    ngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,
                             SpecialTokens(), k=1)
    
    expected = {'i': 0.09090909090909091, 'like': 0.09090909090909091, 
                'a': 0.09090909090909091, 'cat': 0.2727272727272727, 
                'this': 0.09090909090909091, 'dog': 0.09090909090909091, 
                'is': 0.09090909090909091, '<e>': 0.09090909090909091, 
                '<unk>': 0.09090909090909091}

    result = estimate_probabilities(["a"], ngram_model)

    # print("------")
    # print(expected)
    # print(result)
    # print("------")
    assert estimate_probabilities(["a"], ngram_model) == expected, \
      print("estimate_probabilities is not correct")
    
    print("\033[92m Successful test")


estimate_probabilities_test()


# %% [markdown]
# # Question 4: Infer N-Grams

# %%
#@title Q4 Inference

def predict_next_word(sentence_beginning, model):
    """
    Args:
        sentence_beginning: a string
        model: an NGramModel object

    Returns:
        next_word = a string with the next word that his most likely to appear 
        after the sentence_beginning input based ont he model. (You do not need to 
        add in any top K or random sampling.)
        probability = corresponding probability of that word
    """
    # Convert input to lowercase and remove non-alphanumeric characters (except '-')
    sentence_beginning = sentence_beginning.lower()
    sentence_beginning = re.sub(r"[^a-z0-9\-]+", " ", sentence_beginning)

    #tokenize the input sentence
    context_tokens = nltk.word_tokenize(sentence_beginning)
    
    # # get the last (n-1) words as context
    # context_tokens = tokens[-(len(next(iter(model.n_gram_counts.keys()))) - 1):]

    # Get the n-gram window size
    window_size = len(next(iter(model.n_gram_counts.keys())))

    # Ensure context tokens match the required window size
    if not context_tokens:
        context_tokens = [model.special_tokens.start_token]

    if len(context_tokens) >= window_size:
        context_tokens = context_tokens[-window_size:]
    else:
        padding = [model.special_tokens.start_token] * (window_size - len(context_tokens))
        context_tokens = padding + context_tokens

    # get probability distribution for the next word
    probabilities = estimate_probabilities(context_tokens, model)
    # find the word with the hightest probability,with its probability
    next_word = max(probabilities,key=probabilities.get)

    # If predicted word is an end token or unknown token, find the next best word
    if next_word in [model.special_tokens.end_token, model.special_tokens.unknown_token]:
        new_probs = {w: p for w, p in probabilities.items() if w not in [model.special_tokens.end_token, model.special_tokens.unknown_token]}
        if new_probs:
            next_word, probability = max(new_probs.items(), key=lambda x: x[1])

    probability = probabilities[next_word]
    
    return next_word, probability


def predict_next_word_test():
    """
    Test the predict_next_word function.
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
        f.write(tmp_data + '\n')

    # Preprocess data
    sentences, _, vocabulary = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio=1.0)

    # Create n-gram models
    bigram_counts = count_n_grams(sentences, 2, SpecialTokens())
    trigram_counts = count_n_grams(sentences, 3, SpecialTokens())

    ngram_model = NGramModel(bigram_counts, trigram_counts, vocabulary,
                             SpecialTokens(), k=1)

    # Test the function
    next_word, probability = predict_next_word("i like", ngram_model)

    print(f"Predicted next word for 'i like': {next_word} (Probability: {probability:.4f})")


predict_next_word_test()


# %%
# Example Usage
if __name__ == "__main__":
    # Create an instance of your NGramModel here, using your training data
    special_tokens = SpecialTokens()
    count_threshold = 10
    train_data_replaced, test_data_replaced, vocabulary = preprocess_data(
        "en_US.twitter.txt", count_threshold, special_tokens
    )

    # n=2
    unigram_counts = count_n_grams(train_data_replaced, 1, special_tokens)
    bigram_counts = count_n_grams(train_data_replaced, 2, special_tokens)

    ngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,
                            special_tokens, k=1)

    partial_sentence = "i love"  # Example partial sentence
    predicted_word = predict_next_word(partial_sentence, ngram_model)
    print(f"The predicted next word for '{partial_sentence}' is: {predicted_word}")

# %% [markdown]
# # Question 5: Extra Credit: Stylistic N-Grams

# %%

#@title Q5 Extra Credit

class StyleGram:

    def __init__(self, style_files):
        """
        We will only be passing style_files in. All your processing and 
        training should be done by the time this function retunrs.
        """
        self.style_files = style_files
        self.special_tokens = SpecialTokens()
        self.models = []
        self.vocabulary = set()

        # Preprocess and create an NGramModel for each style file
        for file in style_files:
            train_data, _, vocabulary = preprocess_data(file, count_threshold=2, special_tokens=self.special_tokens)
            self.vocabulary.update(vocabulary)  # Accumulate the global vocabulary
            
            bigram_counts = count_n_grams(train_data, 2, self.special_tokens)
            trigram_counts = count_n_grams(train_data, 3, self.special_tokens)

            model = NGramModel(bigram_counts, trigram_counts, vocabulary, self.special_tokens, k=1)
            self.models.append(model)

    def write_in_style_ngram(self, passage):
        """
        Takes a passage in, matches it with a style, given a list of
        filenames, and predicts the next word that will appear
        using a bigram model. 
            
        Args:
            passage: A string that contains a passage
            style_file: a list of filenames to be used to determine the style
            
        Returns:
             single word <string>
             probability associated with the word <float>
             index of "style" it originated from (e.g., 0 for 1st file) <int8>
             probability associated with the style <float>
        """

        
        best_word = None
        best_probability = 0.0
        best_style_index = -1
        best_style_probability = 0.0

        # Iterate over each style model to find the best match
        for index, model in enumerate(self.models):
            next_word, probability = predict_next_word(passage, model)

            if probability > best_probability:
                best_word = next_word
                best_probability = probability
                best_style_index = index
                best_style_probability = probability  # Here, using probability as a style indicator

        return best_word, best_probability, best_style_index, best_style_probability
        # return word, probability_word, style_file, probability_style


# %%
if __name__ == "__main__":
    # Download the datasets
    import urllib.request

    urls = [
        "https://course.ccs.neu.edu/cs6120s25/data/hemingway/hemingway-edit.txt",
        "https://www.gutenberg.org/cache/epub/100/pg100.txt",
        "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"
    ]
    files = ["hemingway.txt", "shakespeare.txt", "dickinson.txt"]

    # Download files
    for url, file in zip(urls, files):
        urllib.request.urlretrieve(url, file)

    # Initialize StyleGram with the downloaded files
    style_model = StyleGram(files)

    # Predict the next word and style
    partial_sentence = "The sun rises"
    word, prob_word, style_index, prob_style = style_model.write_in_style_ngram(partial_sentence)

    print(f"Predicted next word: '{word}' with probability {prob_word:.4f}")
    print(f"Most likely style: {['Hemingway', 'Shakespeare', 'Dickinson'][style_index]}")
    print(f"Style probability: {prob_style:.4f}")



