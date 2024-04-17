import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
            except ValueError as e:
                pass
    return embeddings_index

glove_embeddings = load_embeddings('vector.txt')


def meaning_similarity(word1, word2, embeddings):
    """
    Calculate the meaning similarity between two words using pre-trained word embeddings.
    Returns a value between 0 and 1, where 1 means the words have identical meanings.
    """
    if word1 not in embeddings or word2 not in embeddings:
        return 0  # If either word is not in the embeddings, return 0 similarity

    embedding1 = embeddings[word1].reshape(1, -1)
    embedding2 = embeddings[word2].reshape(1, -1)
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

    return similarity_score