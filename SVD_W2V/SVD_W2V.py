# Assignment5
# Yunpei GU

from collections import Counter
import numpy as np
import re
import pickle
from collections import Counter
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
import urllib.request

def format_time(timestamp):
    """format time"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

# Question 1: Pre-Processing ------------------------

def process_data(filename, min_cnt, max_cnt, min_win = 5, min_letters = 3):
    '''
    Preprocesses and builds the distribution of words in sorted order 
    (from maximum occurrence to minimum occurrence) after reading the 
    file. Preprocessing will include filtering out:
    * words that have non-letters in them,
    * words that are too short (under minletters)
    
    Arguments:
        - filename: name of file
        - min_cnt: min occurrence of words to include
        - max_cnt: max occurrence of words to include
        - min_win: minimum number of words in a title after word filtering
        - min_letters: min length of words to include (3)
    
    Returns:
        - word_freqs: A sorted (max to min) list of tuples of form -
            [(word1, count1), (wordN, countN), ... (wordN, countN)]
        - dataset: A list of strings with OOV words removed -
            ["this is title 1", "this is title 2", ...]
    '''

    # 1. Read the file line-by-line
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 2. Tokenize each line into words.
    tokenized_lines = []

    count = 0
    # threshold = 1_000_000

    for line in lines:
        # Lower-case
        line = line.lower().strip()
        # Split by whitespace
        raw_words = line.split()

        # Keep only words of length >= min_letters and purely alphabetic
        filtered = []
        for w in raw_words:
            # We'll only keep words with [a-z]+ and length >= min_letters
            if len(w) >= min_letters and re.match(r'^[a-z]+$', w):
                filtered.append(w)
        # Store the filtered tokens for now (list of words)
        tokenized_lines.append(filtered)

        # if count==threshold:
        #     break
        count+=1

    print(f"total number of lines: {count}")

    # 3. Count frequencies of all words
    all_words = []
    for tokens in tokenized_lines:
        all_words.extend(tokens)

    counts = Counter(all_words)

    # 4. Build vocab for words meeting frequency constraints [min_cnt, max_cnt]
    vocab = set()
    for w, c in counts.items():
        if c >= min_cnt and c <= max_cnt:
            vocab.add(w)

    # 5. Filter out OOV words from each line, skip if line length < min_win
    final_dataset = []
    for tokens in tokenized_lines:
        in_vocab = [w for w in tokens if w in vocab]
        if len(in_vocab) >= min_win:
            final_dataset.append(" ".join(in_vocab))

    # 6. Create a list of (word, freq) sorted descending by freq
    #    limited to the words that survived the vocab filter
    freq_list = [(w, counts[w]) for w in vocab]
    freq_list.sort(key=lambda x: x[1], reverse=True)

    return freq_list, final_dataset


def plot_distribution(word_freqs):
    """
    Plots the Zipfian distribution of word frequencies.
    """
    frequencies = [count for _, count in word_freqs]
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(frequencies) + 1), frequencies, marker="o", linestyle="none")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Zipfian Distribution of Words in Dataset")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


# Question 2: Matrix Factorization  ------------------------

# Q 2.1: Create an Adjacency Matrix

def create_adjacency(dataset, word2index, win = 10):
    '''
    Builds an adjacency matrix based on word co-occurrence within a window.

    Args:
        - dataset: List of processed titles
        - word2index: Dictionary mapping word to index
        - win: The window size for co-occurrence.

    Returns:
        - adjacency_matrix: A NumPy array representing the adjacency matrix.
    '''
    vocab_size = len(word2index)
    # Use a sparse format for memory efficiency: LIL 
    adj_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    for title in dataset:
        words = title.split()
        # turn words -> indices
        indices = [word2index[w] for w in words if w in word2index]
        # indices = [word2index[w] for w in words ]

        for center_pos in range(len(indices)):
            center_word_idx = indices[center_pos]
            # define the window bounds
            start = max(0, center_pos - win)
            end   = min(len(indices), center_pos + win + 1)
            
            for ctx_pos in range(start, end):
                if ctx_pos == center_pos:
                    continue #  skip the work itself
                ctx_word_idx = indices[ctx_pos]
                adj_matrix[center_word_idx, ctx_word_idx] += 1.0

    # Convert to a more efficient format for SVD: CSR
    adj_matrix = adj_matrix.tocsr()
    return adj_matrix

def build_index_maps(word_freqs):
    """
    Given sorted list [(word, freq), ...], 
    build a dictionary: word2index and index2word
    """
    word2index = {}
    index2word = {}
    for i, (w, f) in enumerate(word_freqs):
        word2index[w] = i
        index2word[i] = w
    return word2index, index2word


def test_create_adjacency():
    test_dataset = [
        "deep learning for natural language processing",
        "language processing and deep learning",
        "natural intelligence and machine learning",
        "machine learning in deep networks"
    ]
    words = sorted(set(word for line in test_dataset for word in line.split()))  # 排序保证一致性
    word2index = {word: i for i, word in enumerate(words)}

    adj_matrix = create_adjacency(test_dataset, word2index, win=2)

    vocab_size = len(word2index)
    assert adj_matrix.shape == (vocab_size, vocab_size), "wrong size of the matrix"

    assert adj_matrix.nnz > 0, "matrix all zeros"

    print("size:", adj_matrix.shape)
    print("non zero size:", adj_matrix.nnz)
    print("5x5: ")
    print(adj_matrix[:5, :5].toarray())



# %%
# Q 2.2: Create SVD Word Vectors --------------------------

def train_svd(adjacency_matrix, min_sv_index = 3, max_sv_index = 103):
    """
    Creates an embedding space using SVD on the adjacency matrix.
    Args:
        adjacency_matrix: The adjacency matrix.
        embedding_dim: The desired dimensionality of the embedding space.
    Returns:
        A NumPy array representing the embedding space (num_words x embedding_dim)
    """
    # 1) Convert to dense if it's a sparse matrix
    dense_mat = adjacency_matrix.toarray()  # May be large!

    # 2) Perform standard SVD
    #    np.linalg.svd returns U, S, Vt with S sorted in descending order.
    #    U.shape = (V, V), S.shape = (V,), Vt.shape = (V, V)
    U, S, Vt = np.linalg.svd(dense_mat, full_matrices=False)

    # 3) Slice out the part we want
    #    By default, S is in descending order already.
    embedding_dim = max_sv_index - min_sv_index

    # min_sv_index : max_sv_index
    S_slice = S[min_sv_index:max_sv_index]        # shape (embedding_dim,)
    U_slice = U[:, min_sv_index:max_sv_index]     # shape (V, embedding_dim)

    # 4) Multiply columns of U by sqrt(S).
    #    E = U_slice * sqrt(S_slice)
    E = U_slice * np.sqrt(S_slice)

    return E

def nearest_neighbors(word, embeddings, word2index, index2word, topk=10):
    """
    Prints out the topk nearest words in 'embeddings' to the given 'word'.
    'embeddings' is assumed shape (vocab_size, embedding_dim).
    We'll do a simple cosine-sim. 
    """
    if word not in word2index:
        print(f"'{word}' not in vocabulary.")
        return
    # find the index of the word
    wid = word2index[word]
    # find the word in the embeddings 
    wvec = embeddings[wid]
    # compute cos-sim with all embeddings
    sims = np.dot(embeddings, wvec)  # shape (vocab_size,)
    
    # decreasing sort
    sorted_indices = np.argsort(-sims) 
    
    # filter out the word itself
    nn_ids = [i for i in sorted_indices if i != wid][:topk]

    # print topk
    print(f"Nearest neighbors for [{word}]:")
    for rank in range(topk):
        cand_id = nn_ids[rank]
        cand_word = index2word[cand_id]
        cand_sim  = sims[cand_id]
        print(f"  {rank+1}. {cand_word} (similarity={cand_sim:.4f})")
    print()


# %%
# Q 3.1: Negative Sampling ------------------------

def sample_w2v(dataset, word2index, neg_samples=5, win=10):
    '''
    Randomly samples:
      - one title (line in dataset)
      - one target word within that title
      - one positive context word from the local +/- win window
      - "neg_samples" negative samples from the vocabulary

    We return the indices: (wi, wo, Wn)
      wi: index of the input/target word
      wo: index of the positive context word
      Wn: list/array of length neg_samples with negative sample indices

    Args:
        dataset:     list of preprocessed titles
        word2index:  map from word to unique integer
        neg_samples: # negative samples
        win:         +/- window size

    Returns:
        (wi, wo, Wn) where each is an integer index in [0, vocab_size).
    '''
    # 1. pick a random line
    # rnd_title = np.random.choice(dataset)
    # speedup by avoiding `np.random.choice`
    rnd_idx = np.random.randint(0, len(dataset))  
    rnd_title = dataset[rnd_idx]

    words = rnd_title.split()
    idx_words = [word2index[w] for w in words]
    length = len(idx_words)

    # 2. pick a random target position
    center_pos = np.random.randint(0, length)
    wi = idx_words[center_pos]  # target word index

    # 3. pick a random context word from the +/- win region 
    start = max(0, center_pos - win)
    end   = min(length, center_pos + win + 1)
    # exclude the center position itself
    ctx_positions = list(range(start, center_pos)) + list(range(center_pos+1, end))

    # edge case: empty context 
    if not ctx_positions:
        wo = wi
    else:
        # ctx_pos = np.random.choice(ctx_positions)
        # wo = idx_words[ctx_pos]
        ctx_pos_idx = np.random.randint(0, len(ctx_positions)) 
        wo = idx_words[ctx_positions[ctx_pos_idx]]

    # 4. pick negative samples from the entire vocabulary (uniformly)
    # NOTE: for very large vocabulary (vocab_size > 50,000), some do a p^0.75 sampling. 
    vocab_size = len(word2index)
    Wn = np.random.randint(0, vocab_size, size=neg_samples)

    return wi, wo, Wn


# Q 3.3: Gradient Implementation ------------------------
# Gradient for Word2Vec with Negative Sampling

def sigmoid(x):
    x = np.clip(x, -20, 20) # Clip x to avoid overflow in exp
    return 1 / (1 + np.exp(-x))

def w2vgrads(vi, vo, vns):
    """
    This function implements the gradient for all vectors in
    input matrix Vi and output matrix Vo.
    Args:
        vi:  Vector of shape (d,), a sample in the input word
            vector matrix
        vo:  Vector of shape (d,), a positive sample in the output
            word vector matrix
        vns: Vector of shape (d, k), k negative samples in the
            output word vector matrix
    Returns:
        dvi, dvo, dVns: the gradients of J with respect to vi and vo.
    """

    # pos sample: σ(vi · vo)
    pos_score = sigmoid(np.dot(vi, vo)) 

    # neg sample: σ(-vi · vn)
    neg_score = sigmoid(-np.dot(vi, vns))  # shape (k,)

    # dJ/dvi = (1 - σ(vi · vo)) * vo - sum_{vn} [(1 - σ(-vi · vn)) * vn]
    dvi_pos = (1.0 - pos_score) * vo
    dvi_neg = np.dot((1.0-neg_score), vns.T)
    dvi = dvi_pos - dvi_neg

    # dJ/dvo:
    #    = (1 - p)*vi
    dvo = (1.0 - pos_score) * vi

    # dJ/dVns
    dVns = - (1.0 - neg_score) * vi[:, np.newaxis]

    return dvi, dvo, dVns

# Q 3.4: Create W2V Embeddings ------------------------
# Training the Word2Vec Vectors

def train_w2v(dataset, word2index, iters=100000, negsamps=5,
              win=5, embedding_dim=100, learning_rate=0.01,
              ):
    """
    Creates an embedding space using negative-sampling Word2Vec.

    We'll store 2 sets of vectors: 
       - Vi: shape (vocab_size, d) => "input" vectors
       - Vo: shape (vocab_size, d) => "output" vectors

    On each iteration:
      1) sample (wi, wo, Wn)
      2) retrieve vi = Vi[wi], vo = Vo[wo], vns = Vo[Wn] 
      3) compute gradient wrt these from w2vgrads
      4) gradient-ascent update

    We track the partial objective from the positive portion, i.e. log(sigma(vi·vo)) 
    to get a sense of how training is going. This is a quick measure of whether 
    we are moving in the right direction.

    Args:
        dataset:       list of processed titles
        word2index:    map from word -> integer index
        iters:         number of total training iterations
        negsamps:      number of negative samples
        win:           window size
        embedding_dim: dimension of each word vector
        learning_rate: step-size for gradient ascent

    Returns:
        V_w2v:  the "input" embeddings, shape (vocab_size, embedding_dim)
        losses: list of recorded partial losses for the positive term only
    """
    vocab_size = len(word2index)

    # Randomly initialize the two sets of vectors
    #  Vi is the "input" embedding matrix
    #  Vo is the "output" embedding matrix
    rng = np.random.default_rng(seed=42)
    Vi = 0.01 * rng.standard_normal(size=(vocab_size, embedding_dim)) #目标词的词向量
    Vo = 0.01 * rng.standard_normal(size=(vocab_size, embedding_dim)) #上下文词的词向量。

    # We'll store partial losses occasionally
    losses = []

    for i in range(int(iters)):
        wi, wo, Wn = sample_w2v(dataset, word2index, neg_samples=negsamps, win=win)

        # retrieve the relevant vectors
        vi = Vi[wi]         # shape (d,)
        vo = Vo[wo]         # shape (d,)
        vns = Vo[Wn].T      # shape (d, k), we transpose so that the columns are negative vectors

        # compute gradients
        dvi, dvo, dVns = w2vgrads(vi, vo, vns)

        # gradient-ascent update
        Vi[wi] += learning_rate * dvi
        Vo[wo] += learning_rate * dvo

        # negative samples
        # Wn is a list of indices, shape (k,)
        # dVns is (d,k), so the j-th negative sample index is Wn[j]
        for j, idxn in enumerate(Wn):
            Vo[idxn] += learning_rate * dVns[:, j]

        # occasionally store the positive portion of the objective:
        if (i+1) % 1000 == 0:
            # positive score = sigma(vi dot vo)
            dotpos = np.dot(vi, vo)
            # dotpos = np.dot(Vi[wi], Vo[wo])
            posval = np.log(1.0 / (1.0 + np.exp(-dotpos)) + 1e-10)
            # posval = 1.0 / (1.0 + np.exp(-np.clip(dotpos, -10, 10)))  # 限制范围 [-10,10]

            negval = 0
            for j in range(vns.shape[1]):
                dotneg = -np.dot(vi, vns[:, j])
                negval += np.log(1.0 / (1.0 + np.exp(-dotneg)) + 1e-10)

            # store log(sigma(vi dot vo)) = log(posval)
            losses.append(posval + negval)

            # print(dotpos, posval, posval + negval)

    # Return the "input" vectors as the final word embeddings
    return Vi, losses


import pickle

def save_model(word_freqs, V_svd1, V_svd2, V_w2v1, V_w2v2, filename="assignment5.pkl"):
    """
    Saves the model parameters into a pickle file.

    Args:
        word_freqs: Processed word frequency distribution.
        V_svd: SVD-based word embeddings.
        V_w2v: Word2Vec embeddings.
        filename: Name of the file to save the model.
    """
    data = {
        'word_freqs': word_freqs,
        'V_svd1': V_svd1,
        'V_svd2': V_svd2,
        'V_w2v1': V_w2v1,
        'V_w2v2': V_w2v2,
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Model parameters saved to {filename}")


# -----------------------------main-----------------------------

# Dataset URL
dataset_url = "https://course.ccs.neu.edu/cs6120s25/data/arxiv-titles.txt"
dataset_filename = "arxiv_titles.txt"

# Check if the dataset exists, if not, download it
if not os.path.exists(dataset_filename):
    print(f"Downloading dataset from {dataset_url}...")
    urllib.request.urlretrieve(dataset_url, dataset_filename)
    print(f"Dataset saved as {dataset_filename}")
else:
    print(f"Dataset {dataset_filename} already exists.")


# Question 1: Pre-Processing -----------------------------
min_cnt = 200
max_cnt = 138000 # exclude the word "from"
min_win=5
min_letters=3

print(f"min_cnt: {min_cnt}\n"
      f"max_cnt: {max_cnt}\n"
      f"min_win: {min_win}\n"
      f"min_letters: {min_letters}\n")

word_freqs, dataset = process_data(
    filename="arxiv_titles.txt",
    min_cnt=min_cnt,
    max_cnt=max_cnt,
    min_win=min_win,
    min_letters=min_letters
)
plot_distribution(word_freqs)

print(f"Number of lines (titles) after filtering = {len(dataset)}")
print(f"Vocabulary size = {len(word_freqs)}")

# Build the dictionary for adjacency / SVD / Word2Vec 
word2index, index2word = build_index_maps(word_freqs)

print(f"number of titles: {len(dataset)}")


# Q2.1: Create adjacency matrix-----------------------------

start_time = time.time()
adj_matrix = create_adjacency(dataset, word2index, win=10) #6min 30sc
end_time = time.time()

print("Adjacency matrix shape:", adj_matrix.shape)
print(f"Time taken to create adjacency matrix: {end_time - start_time:.4f} seconds")
# 579.2750 seconds

# Filter out low-frequency co-occurrences (keep matrix sparsity)
# Values below the threshold (3 occurrences) are set to 0.
adj_matrix.data[adj_matrix.data < 3] = 0  

# Apply log transformation to smooth frequency values
# Use `log1p(x) = log(1 + x)` instead of `log(x)` to avoid `log(0)`.
adj_matrix.data = np.log1p(adj_matrix.data)  

# normalization (use L1)
row_sums = np.array(adj_matrix.sum(axis=1)).flatten() 
row_sums[row_sums == 0] = 1  # Prevent division by zero
adj_matrix = adj_matrix.multiply(1 / row_sums[:, np.newaxis])  # 逐行归一化



# Q2.2: Train SVD ------------------------------------------------------------
print("-------- Train SVD --------")
min_sv_index = 0
max_sv_index = 100
print(f"min_sv_index: {min_sv_index}\n"
      f"max_sv_index: {max_sv_index}\n")

start_time = time.time()
V_svd1 = train_svd(adj_matrix, min_sv_index=min_sv_index, max_sv_index=min_sv_index)
end_time = time.time()
print("SVD-based embeddings shape:", V_svd1.shape)
print(f"Time taken to compute SVD: {end_time - start_time:.4f} seconds")

V_svd1_normalized = V_svd1 / (1e-9 + np.linalg.norm(V_svd1, axis=1, keepdims=True))

start_time = time.time()
# print(f"{format_time(start_time)} nearest_neighbors start")
for test_word in ["neural", "machine", "dark", "string", "black"]:
    nearest_neighbors(test_word, V_svd1_normalized, word2index, index2word, topk=10)
end_time = time.time()

print(f"Time taken to find nn: {end_time - start_time:.4f} seconds\n")



# Q2.2: Train SVD ------------------------------------------------------------
min_sv_index = 5 # value close to 0 may be non-sense high freq occurance e.g. with 'the' 'is'
max_sv_index = 105
print(f"min_sv_index: {min_sv_index}\n"
      f"max_sv_index: {max_sv_index}\n")
start_time = time.time()
V_svd2 = train_svd(adj_matrix, min_sv_index=min_sv_index, max_sv_index=max_sv_index)
end_time = time.time()
print("SVD-based embeddings shape:", V_svd2.shape)
print(f"Time taken to compute SVD: {end_time - start_time:.4f} seconds")

V_svd2_normalized = V_svd2 / (1e-9 + np.linalg.norm(V_svd2, axis=1, keepdims=True))

start_time = time.time()
# print(f"{format_time(start_time)} nearest_neighbors start")
for test_word in ["neural", "machine", "dark", "string", "black"]:
    nearest_neighbors(test_word, V_svd2_normalized, word2index, index2word, topk=10)
end_time = time.time()
# print(f"{format_time(end_time)} nearest_neighbors end")
print(f"Time taken to find nn: {end_time - start_time:.4f} seconds\n")



# Question 3: Word2Vec ------------------------------------------------------------
iters=3_000_000      
negsamps=5
win=5
embedding_dim=100 
learning_rate=0.01

print(f"iters: {iters}\n"
      f"negsamps: {negsamps}\n"
      f"win: {win}\n"
      f"embedding_dim: {embedding_dim}\n"
      f"learning_rate: {learning_rate}")

start_time = time.time()
print(f"{format_time(start_time)} train_w2v start")
V_w2v1, losses = train_w2v(
    dataset, 
    word2index, 
    iters=iters, 
    negsamps=negsamps, 
    win=win, 
    embedding_dim=embedding_dim, 
    learning_rate=learning_rate,
)

end_time = time.time()
print("Word2Vec embeddings shape:", V_w2v1.shape)
print(f"Time taken to compute w2v: {end_time - start_time:.4f} seconds")
# Smoothed loss curve (averaged over 20 checkpoints)
losses_smooth = [sum(losses[i*100:i*100+100]) / 100 for i in range(len(losses) // 100)]
# x_values_smooth = np.arange(20_000, len(losses) * 1000 + 1, 20_000)  # Every 20 checkpoints
x_values_smooth = np.arange(100_000, len(losses) * 1000 + 1, 100_000)  # Every 100 checkpoints

plt.figure()
plt.title(f"Smoothed Word2Vec Training Loss (iters = {iters})")
plt.plot(x_values_smooth, losses_smooth, linestyle="-")
plt.xlabel("Iterations")
plt.ylabel("log(sigmoid(vi·vo))")

plt.gca().invert_yaxis()  # Invert Y-axis if needed
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

# Normalize Word2Vec embeddings for cosine similarity
V_w2v1_normalized = V_w2v1 / (1e-9 + np.linalg.norm(V_w2v1, axis=1, keepdims=True))

# Find nearest neighbors for selected words
for test_word in ["neural", "machine", "dark", "string", "black"]:
    nearest_neighbors(test_word, V_w2v1_normalized, word2index, index2word, topk=10)


iters=4_000_000      
negsamps=5
win=5
embedding_dim=100 
learning_rate=0.008

print(f"iters: {iters}\n"
      f"negsamps: {negsamps}\n"
      f"win: {win}\n"
      f"embedding_dim: {embedding_dim}\n"
      f"learning_rate: {learning_rate}")

start_time = time.time()
print(f"{format_time(start_time)} train_w2v start")
V_w2v2, losses = train_w2v(
    dataset, 
    word2index, 
    iters=iters, 
    negsamps=negsamps, 
    win=win, 
    embedding_dim=embedding_dim, 
    learning_rate=learning_rate,
)

end_time = time.time()
print("Word2Vec embeddings shape:", V_w2v2.shape)
print(f"Time taken to compute w2v: {end_time - start_time:.4f} seconds")
# Smoothed loss curve (averaged over 20 checkpoints)
losses_smooth = [sum(losses[i*100:i*100+100]) / 100 for i in range(len(losses) // 100)]
# x_values_smooth = np.arange(20_000, len(losses) * 1000 + 1, 20_000)  # Every 20 checkpoints
x_values_smooth = np.arange(100_000, len(losses) * 1000 + 1, 100_000)  # Every 100 checkpoints

plt.figure()
plt.title(f"Smoothed Word2Vec Training Loss (iters = {iters})")
plt.plot(x_values_smooth, losses_smooth, linestyle="-")
plt.xlabel("Iterations")
plt.ylabel("log(sigmoid(vi·vo))")

plt.gca().invert_yaxis()  # Invert Y-axis if needed
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

# Normalize Word2Vec embeddings for cosine similarity
V_w2v2_normalized = V_w2v2 / (1e-9 + np.linalg.norm(V_w2v2, axis=1, keepdims=True))

# Find nearest neighbors for selected words
for test_word in ["neural", "machine", "dark", "string", "black"]:
    nearest_neighbors(test_word, V_w2v2_normalized, word2index, index2word, topk=10)


# save embeddings
save_model(word_freqs, V_svd1, V_svd2, V_w2v1, V_w2v2)


