import re
from collections import Counter
import streamlit as st

# Instruction:

# 1/ Ensure you have installed streamlit in your environment:
#     pip install streamlit

# 2/ Ensure the shakespeare-edit.txt file is in the same directory.

# 3/ To start the server, enter:
#     python -m streamlit run assignment1.py

# 4/ you can enter the prefix you like and press enter to get the 10 most frequent words in the texts starting with your prefix.


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
        self.frequency = 0  # store frequency of the word

def process_data(word_list):
    """
    Builds any data structure or model from a list of words.
    Args:
    words: A list of words.
    Returns:
    Any_data_structure_you_like
    2
    """
    model_or_data_structure = TrieNode()

    for rank, word in enumerate(word_list):
        node = model_or_data_structure
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = len(word_list) - rank  

    return model_or_data_structure
   
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
    # Web App
    st.title("Shakespeare Autocomplete App")
    st.write("Enter a prefix you like to see autocomplete suggestions in Shakespearean style.")

    # Load and process data (execute once)
    @st.cache_data
    def load_data():
        filename = "shakespeare-edit.txt" 
        vocabulary = read_vocabulary(filename)
        trie_root = process_data(vocabulary)
        return trie_root

    trie_root = load_data()

    # Input field for prefix
    prefix = st.text_input("Enter the prefix you like:")

    # Display autocomplete suggestions
    if prefix:
        suggestions = autocomplete_word(prefix, trie_root)
        if suggestions:
            st.write("Autocomplete Suggestions:")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
        else:
            st.write("No suggestions found.")
