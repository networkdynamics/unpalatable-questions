"""
This file contains functions to compute Writing-Style features.
"""

import nltk
import re
import numpy as np
from string import punctuation

def get_word_count(words):
    """
    Counts the number of words ignoring punctuation marks.
    """
    return sum(1 for word in words if word not in punctuation)


def get_URL_count(text):
    """
    Parameters
    ----------
    text : str
        comment text

    Returns
    -------
    int
        number of URLs in the comment.
    """
    return len(re.findall(r'(https?:|www\.)', text))


def get_second_person_pronouns(words):
    """
    Parameters
    ----------
    words : list of words in the comment

    Returns
    -------
    int
        number of second person pronouns in the comment (eg: 'you' 'your' 'yours').
    """
    _2nd_prons = set(['you', 'your', 'yours', 'yourself', 'yourselves', 'u', 'ur', 'urs', 'urself'])
    return sum(1 for word in words if word.lower() in _2nd_prons)


def get_question_exclamation_marks(text):
    """
    Parameters
    ----------
    text : str
        comment text

    Returns
    -------
    int
        number of question marks and exclamation marks in the comment.
    """
    puncts = set(['?', '!'])
    return sum([1 for char in text if char in puncts])


def get_capital_words(words):
    """
    Parameters
    ----------
    words : list of words in the comment

    Returns
    -------
    int
        number of capital words in the comment.
    """
    return sum(1 for word in words if word.isupper())


def writingstyle_vector(comment):
    """
    Returns a Writing-Style vector for the given comment.

    Parameters
    ----------
    comment: str (comment text)

    Returns
    -------
    list of 5 features
        Order: [word_count, url_count, pronoun_count, exclamation/question_count, capital_word_count]
    """
    words = nltk.word_tokenize(comment)
    x1 = get_word_count(words)
    x2 = get_URL_count(comment)
    x3 = get_second_person_pronouns(words)
    x4 = get_question_exclamation_marks(comment)
    x5 = get_capital_words(words)
    return [x1, x2, x3, x4, x5]


def style_vectorize(main_sentences, context_sentences, use_comments):
    """
    Vectorizes the given list of comments using the writing style features.
    Can choose to use Comment_Text (will be concatenate at the end).

    Parameters
    ----------
    main_sentences : list of reply sentences (or optionally just 'question' sentences)
    context_sentences : list of comment (context) sentences
    use_comments : boolean (True if cotext comments are to be considered)

    Returns
    -------
    numpy 2D array: vectorized input 'X'
    """
    X_reply = []
    for sent in main_sentences:
        X_reply.append(writingstyle_vector(sent))

    X_reply = np.array(X_reply)

    if use_comments:
        print("Using Comment Text as well....")
        X_comment = []
        for sent in context_sentences:
            X_comment.append(writingstyle_vector(sent))

        X_comment = np.array(X_comment)
        X = np.concatenate((X_reply, X_comment), axis=1)

    else:
        X = X_reply

    return X
