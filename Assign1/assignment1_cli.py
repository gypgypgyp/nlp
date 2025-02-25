import re
from collections import Counter

'''
Instruction:

In your terminal, enter:

    python assignment1_cli.py

to start the app. you can enter the prefix you like and press enter to get the 10 most frequent words in the texts starting with your prefix.
'''


def read_vocabulary(filename):
    """
    Reads in a given file specified by "filename" and processes it
    by removing punctuation, forcing lowercase, splits into
    individual words, and removes the numbers that might appear in
    the text.
    Args:
    filename: the name of the file to be processed
    Returns:
    A list of words in the order in which they appeared in the
    text.
    """
    # read file
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # remove marks and nums, tolower
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # split the words by white space
    words = text.split()

    # count the frequency of words
    word_counts = Counter(words)

    # return the frequency list
    sorted_words = [word for word, _ in word_counts.most_common()]
    return sorted_words



class TrieNode:
    """
    A Trie node to store each character in the word and its children.
    """
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0  # Frequency of the word

def process_data(word_list):
    """
    Builds any data structure or model from a list of words.
    Args:
    words: A list of words.
    Returns:
    Any_data_structure_you_like
    2
    """
    root = TrieNode()

    for rank, word in enumerate(word_list):
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = len(word_list) - rank  # Assign frequency based on rank

    return root
   
    return model_or_data_structure

def autocomplete_word(prefix, model_or_data_structure):
    """
    Returns a list of words starting with the given prefix. This list
    is sorted in order of the frequency (probability) of the words.
    Args:
    prefix: The prefix to search for.
    model_or_data_structure: model or data, considering the frequency
    of word occurrence.
    Returns:
    A list of ten most-common words starting with the prefix.
    """
    node = model_or_data_structure

    # Navigate the Trie to the end of the prefix
    for char in prefix:
        if char not in node.children:
            return []  # If prefix not found, return an empty list
        node = node.children[char]

    # Use DFS to find all words starting from this node
    results = []

    def dfs(current_node, path):
        if current_node.is_end_of_word:
            results.append((''.join(path), current_node.frequency))
        for char, child_node in current_node.children.items():
            dfs(child_node, path + [char])

    dfs(node, list(prefix))

    # Sort results by frequency in descending order and return top 10
    results.sort(key=lambda x: -x[1])
    return [word for word, _ in results[:10]]

# Testing the implementation
if __name__ == "__main__":
    # File path to the Shakespeare dataset
    filename = "shakespeare-edit.txt"
    # Read and process the vocabulary
    vocabulary = read_vocabulary(filename)
    print(f"Vocabulary processed. Total unique words: {len(vocabulary)}")
    # Build the data structure
    trie_root = process_data(vocabulary)
    print("Trie data structure built.")
    # Test autocomplete functionality
    while True:
        prefix = input("\nEnter a prefix (or type 'exit' to quit): ").strip().lower()
        if prefix == "exit":
            break
        suggestions = autocomplete_word(prefix, trie_root)
        if suggestions:
            print(f"Suggestions for '{prefix}': {', '.join(suggestions)}")
        else:
            print(f"No suggestions found for '{prefix}'.")



# filename = "/Users/guyunpei/Downloads/CS6120/Assign1/shakespeare-edit.txt"
# try:
#     with open(filename, 'r', encoding='utf-8') as file:
#         print(read_vocabulary(filename))
#         print("File read successfully!")
# except FileNotFoundError:
#     print("File not found! Check the file path.")