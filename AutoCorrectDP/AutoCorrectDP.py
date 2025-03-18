
import re
from collections import Counter
import string
import numpy as np

def process_data(file_name):
    """
    Input:
        filename: A file_name which is found in your current
                  directory. You just have to read it in.
    Output:
        wordprobs: a dictionary where keys are all the processed
                    lowercase words and the values are the frequency
                    that it occurs in the corpus (text file you read).
    """
    # read file
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # tolower
    text = text.lower()

    # only keep the lower characters
    text = re.sub(r'[^a-z\s]', '', text)

    # split the words by white space
    words = text.split()

    # count the frequency of words
    word_counts = Counter(words)

    # calculate total words
    total_words = sum(word_counts.values())

    # calculate the frequency of every word
    wordprobs = {word: count/total_words for word, count in word_counts.items()}

    return wordprobs


# test process_data

# filename = "shakespeare-edit.txt"

# try:
#     print(process_data(filename))

# except FileNotFoundError:
#     print("File not found! Check the file path.")


# # Question 2: Identifying Probable Words

def probable_substitutes(word, probs, maxret = 10):
    """
    Input:
        word - The misspelled word
        probs - A dictionary of word --> prob
        maxret - Maximum number of words to return
    Returns:
        [(word1, prob1), ... ]
    """
    def delete_letter(word): 
    # given a word, we can change it by removing one character.
        return [word[:i]+word[i+1:] for i in range(len(word))]

    def switch_letter(word): 
    # given a word, we can change it by switching two adjacent characters.
        return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]
    
    def replace_letter(word): 
    # given a word, we can replace one character by another different letter.
        return [word[:i] + c + word[i+1:] for i in range(len(word)) for c in string.ascii_lowercase]

    def insert_letter(word): 
    # given a word, we can insert an additional character.
        return [word[:i] + c + word[i:] for i in range(len(word)+1) for c in string.ascii_lowercase]


    one_edit_words = set(delete_letter(word) + 
                     switch_letter(word) +
                     replace_letter(word) +
                     insert_letter(word)
                    )
    
    candidates = set()

    for w in one_edit_words:
        candidates.update(
            set(delete_letter(w) + 
                     switch_letter(w) +
                     replace_letter(w) +
                     insert_letter(w)
                    )
        )

    candidates.update(one_edit_words)

    valid_candidates = {w:probs[w] for w in candidates if w in probs and w != ""}

    return sorted(valid_candidates.items(), key = lambda x: x[1], reverse = True)[:maxret]


# test probable_substitutes

# corpus_probs = {
#     "learning": 0.02, "leaning": 0.005, "turning": 0.001,
#     "earning": 0.003, "leaping": 0.0008, "burning": 0.0005
# }

# word = "lerningg"

# suggestions = probable_substitutes(word, corpus_probs)

# print(suggestions)


# # Question 3: Computing the Minimum Edit Distance


def min_edit_distance(source, target, ins_cost = 1,
                      del_cost = 1, rep_cost = 2):
    ''' 
    Input:
        source: starting string
        target: ending string
        ins_cost: integer representing insert cost
        del_cost: integer representing delete cost
        rep_cost: integer representing replace cost
    Output:
        D: matrix of size (len(source)+1 , len(target)+1)
           with minimum edit distances
        med: the minimum edit distance required to convert
             source to target
    '''

    m,n = len(source), len(target)

    # initialize the dp array
    DP = np.zeros((m+1,n+1), dtype = int)

    # initialize first col (source --> "")
    for i in range(1,m+1):
        DP[i][0] = DP[i-1][0] + del_cost

    # initialize first row ("" --> target)
    for j in range(1,n+1):
        DP[0][j] = DP[0][j-1] + ins_cost
    
    # fill the DP array
    for i in range(1,m+1):
        for j in range(1,n+1):
            # if the current character is the same, there is no cost
            if source[i-1] == target[j-1]:
                cost = 0
            else:
                cost = rep_cost
            
            # get the min edit cost
            DP[i][j] = min(DP[i-1][j] + del_cost,
                           DP[i][j-1] + ins_cost,
                           DP[i-1][j-1] + cost)
    
    return DP, DP[-1][-1]

# test min_edit_distance

# source = "waht"
# target = "what"
# DP, med = min_edit_distance(source, target)

# for x in DP:
#     print(x)

# print(f"\nMinimum Edit Distance between '{source}' and '{target}': {med}")

if __name__ == "__main__":
    filename = "shakespeare-edit.txt"
    word_probs = process_data(filename)
    print("Loaded corpus with", len(word_probs), "vocabularies.")
    top_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 most frequent words:", top_words)

    test_word = "lerningg"
    suggestions = probable_substitutes(test_word, word_probs)
    print("Suggestions for", test_word, ":", suggestions)
    
    source_word = "waht"
    target_word = "what"
    _, edit_distance = min_edit_distance(source_word, target_word)
    print(f"Minimum edit distance from '{source_word}' to '{target_word}':", edit_distance)





