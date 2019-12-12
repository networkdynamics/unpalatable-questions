"""
This file contains functions to compute lexicon-based features.
"""

import pandas as pd
import enchant
import nltk
from nltk.util import ngrams
from empath import Empath
from string import punctuation

def load_hate_lexicon():
    """
    Loads all the unigrams, bigrams, and trigrams in our hate lexicon. See README in /hate-lexicon/ for more details.
    """
    lexicon_path = './hate-lexicon/'

    a1 = pd.read_csv(lexicon_path+'fatpeoplehate-manually-filtered-hate-words.csv', header=None).iloc[:,0].tolist()
    a2 = pd.read_csv(lexicon_path+'coontown-manually-filtered-hate-words.csv', header=None).iloc[:,0].tolist()

    with open(lexicon_path+'wydl.txt', 'r') as f:
        lines = f.read().splitlines()
    a3 = []
    for row in lines:
        word = row.split(':')[0]
        a3.append(word.replace('"', '').rstrip())

    temp = pd.read_csv(lexicon_path+'hatebase_dict.csv', header=None).iloc[:,0].tolist()
    a4 = []
    for w in temp:
        a4.append(w.replace("'", '').replace(",", ''))

    a5 = pd.read_csv(lexicon_path+'refined_ngram_dict.csv', header=None).iloc[:,0].tolist()

    with open(lexicon_path+'my_lexicon.txt', 'r') as f:
        a6 = f.read().splitlines()

    with open(lexicon_path+'badwords.txt', 'r') as f:
        a7 = f.read().splitlines()

    temp = a1 + a2 + a3 + a4 + a5 + a6 + a7
    hate_lexicon = set([x.lower() for x in temp])
    print("There are {} words in our hate lexicon.".format(len(hate_lexicon)))
    return hate_lexicon


def get_hate_word_count(comment):
    """
    Counts the number of hate words/bigrams/trigrams (from our lexicon) present in the comment.

    Parameters
    ----------
    comment: str

    Returns
    -------
    int
        Count of the number of hate words
    """
    words = nltk.word_tokenize(comment)
    count = 0
    for n in [1,2,3]:
        for tup in ngrams(words, n):
            temp = '' # temp is the unigram/bigram/trigram
            for word in tup:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in HATE_LEXICON: # check if it exists in lexicon
                count += 1
    return count


def get_non_english_word_count(comment):
    """
    Counts the number of non-english words present in the comment. Determines non-English by comparing against all NLTK words, pyenchant dictionary, and additional checks.
    Note: still possibly fails on entity  names.

    Parameters
    ----------
    comment: str

    Returns
    -------
    int
        Number of words that do not exist in nltk.corpus.words (proxy for English Dictionary).
        Note: Ignores URLs and punctuation marks.
    """
    words = nltk.word_tokenize(comment)
    count = 0
    for word in words:
        word = word.lower()
        if word.startswith('www.') or word.startswith('http') or word.endswith('.com'): # skip URLs
            continue
        if word not in ALL_NLTK_WORDS: # count words that don't exist in ALL_NLTK_WORDS
            if ENCHANT_DICT.check(word) == False: # Confirm with pyenchant
                count += 1
    return count


def lexicon_vector(text):
    """
    Returns a Lexicon vector for the given comment.

    Parameters
    ----------
    text: str (comment text)

    Returns
    -------
    list of 17 features
        Order: [hate_count, non_eng_count]+[empath vector]
    """
    x1 = get_hate_word_count(text)
    x2 = get_non_english_word_count(text)
    x3 = empath_vector(text)
    return [x1, x2] + x3


def empath_vector(text):
    """
    Returns a normalised vector (list) of 15 hand-picked categories from Empath: http://empath.stanford.edu/
    """
    categories = ['hate', 'aggression', 'dispute', 'swearing_terms', 'ridicule', 'exasperation', 'fight', 'politeness', 'disgust', 'rage', 'warmth', 'sadness', 'shame', 'negative_emotion', 'positive_emotion']
    lex = Empath()
    d = lex.analyze(text, categories=categories, normalize=True)
    if d == None:
        return 15*[0.0]
    return list(d.values())

HATE_LEXICON = load_hate_lexicon()
ALL_NLTK_WORDS = set(nltk.corpus.words.words() + list(punctuation) + ["'s", "n't", "'re", "'m", "'ve", "'d"]) # preparing a list of all words
ENCHANT_DICT = enchant.Dict("en_US")
