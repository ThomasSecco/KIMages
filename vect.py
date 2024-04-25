import spacy

# Load the English language model
nlp = spacy.load("en_core_web_md")

def simi(word1,word2):
    doc = nlp(word1)
    doc2 = nlp(word2)
    return doc.similarity(doc2)


