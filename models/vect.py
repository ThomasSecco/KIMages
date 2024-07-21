import spacy

# Load the medium-sized English language model from spaCy.
nlp = spacy.load("en_core_web_md")

def simi(word1, word2):
    """
    Calculate the semantic similarity between two words using spaCy's language model.

    Parameters:
        word1 (str): The first word to compare.
        word2 (str): The second word to compare.

    Returns:
        float: The similarity score between the two words, ranging from 0 (no similarity) to 1 (identical).
    """
    # Convert the first word into a spaCy document object
    doc = nlp(word1)
    # Convert the second word into a spaCy document object
    doc2 = nlp(word2)
    # Use spaCy's built-in method to compute the similarity between the two document objects
    return doc.similarity(doc2)
