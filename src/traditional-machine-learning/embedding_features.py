"""
This file contains functions to computed features related to word embedding models.
"""
import numpy as np
import gensim
import io
import nltk

def calculate_vector(model, text, dimensionality=300):
    """
    Calculates the average vector for the given text, and the word-embedding model.

    Parameters
    ----------
    model : word embedding model
    text : str (comment or reply text to be converted to a vector)
    dimensionality : dimensionality of vectors (default=300)

    Returns
    -------
    list of floats
        Average vector for the given text
    """
    try:
        words = nltk.word_tokenize(unicode(text, errors='ignore'))
    except:
        words = nltk.word_tokenize(text)

    # Filter words that are not present in our word-embedding model
    my_words = []
    for w in words:
        if w in model:
            my_words.append(w)

    # Average vectors of each word:
    vector = [0.0]*dimensionality
    number_of_words = len(my_words)

    # If none of the words are in our model, return a null vector: vector of all 0s.
    if number_of_words == 0:
        return vector

    # Else, average:
    for w in my_words:
        vec = model[w]
        vector = np.add(vector, vec)
    avg_vector = np.nan_to_num(vector/number_of_words)
    return avg_vector


def embedding_vectorize(model, main_sentences, context_sentences, use_comments):
    """
    Vectorizes the text using the pre-trained embedding model. Can choose to use Comment_Text (will be concatenate at the end).

    Parameters
    ----------
    model : word embedding model
    main_sentences : list of reply sentences (or optionally just 'question' sentences)
    context_sentences : list of comment (context) sentences
    use_comments : boolean (True if cotext comments are to be considered)

    Returns
    -------
    numpy 2D array: vectorized input 'X'
    """
    dimensionality = 300 # Dimensionality is 300 for all three pre-trained models and Reddit-trained model.

    X_reply = []
    for sent in main_sentences:
        X_reply.append(calculate_vector(model, sent, dimensionality))

    X_reply = np.array(X_reply)

    if use_comments:
        print("Using Comment Text as well....")
        X_comment = []
        for sent in context_sentences:
            X_comment.append(calculate_vector(model, sent, dimensionality))

        X_comment = np.array(X_comment)
        X = np.concatenate((X_reply, X_comment), axis=1)

    else:
        X = X_reply

    return X


# Functions to load the embedding models:
def load_Fasttext_vectors(fname):
    """
    Loads Fasttext vectors given the file path.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())

    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def loadGloveModel(gloveFile): # Thanks to: https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    """
    Loads the GloVe vectors given the file path.
    """
    embedding_dict = {}
    f = open(gloveFile, 'r', encoding='UTF-8')

    for line in f.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        embed_vector = np.array([float(i) for i in row[1:]]) # convert to list of float
        embedding_dict[vocab_word]=embed_vector

    f.close()
    return embedding_dict


def load_embedding_models():
    """
    Loads and returns the four embedding models.
    """
    path = '/path/word_embeddings/'
    # Loading Google's pre-trained word2vec:
    google_vec = gensim.models.KeyedVectors.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', \
                                                                 binary=True)
    print("Google's Word2Vec Loaded | Type: {}".format(type(google_vec)))
    # word2vec obviously does not cover some words, surprising ones: "a", "and", "of", "to"

    # Loading Fasttext pre-trained:
    fast_text_vec = load_Fasttext_vectors(path+'crawl-300d-2M.vec') # dict with key=word and value=vector
    print("Fasttext Loaded | Type: {}".format(type(fast_text_vec)))

    # Loading GloVe pre-trained:
    glove_vec = loadGloveModel(path+'glove.840B.300d.txt')
    print("GloVe Loaded | Type: {}".format(type(glove_vec)))

    # Loading Reddit-trained word2vec:
    reddit_vec = gensim.models.Word2Vec.load(path+'reddit_trained_allof2016.word2vec')
    print("Reddit Word2Vec Loaded | Type: {}".format(type(reddit_vec)))

    return google_vec, fast_text_vec, glove_vec, reddit_vec
