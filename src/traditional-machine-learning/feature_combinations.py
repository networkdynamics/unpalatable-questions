"""
This file has functions to compute feature vectors for different combinations of feature categories.
"""

import nltk
import gensim
import numpy as np
import writing_style_features
import lexicon_features
import embedding_features
import sentiment_features


def ws_lexicon_vector(comment):
    """
    Returns a Writing-Style + Lexicon vector for the given comment.

    Parameters
    ----------
    comment: str (comment text)

    Returns
    -------
    list of 7 features
        Order: [word_count, url_count, pronoun_count, exclamation/question_count, capital_word_count, hate_count, non_eng_count]
    """
    w = writing_style_features.writingstyle_vector(comment)
    l = lexicon_features.lexicon_vector(comment)
    return w+l


def embedding_ws_lexicon_vector(comment):
    """
    Returns a Word-Embedding + Writing-Style + Lexicon vector for the given comment.

    Parameters
    ----------
    comment: str (comment text)

    Returns
    -------
    list of 307 features
        Order: [word_embedding] + [word_count, url_count, pronoun_count, exclamation/question_count, capital_word_count, hate_count, non_eng_count]
    """
    a = embedding_features.calculate_vector(model=GLOVE_VEC, text=comment)
    b = ws_lexicon_vector(comment)
    vec = list(a) + list(b)
    return vec


def sentiment_ws_lexicon_vector(comment):
    """
    Returns a VADER-Sentiment + Writing-Style + Lexicon vector for the given comment.

    Parameters
    ----------
    comment: str (comment text)

    Returns
    -------
    list of 10 features
        Order: [pos_score, neg_score, neu_score, word_count, url_count, pronoun_count, exclamation/question_count, capital_word_count, hate_count, non_eng_count]
    """
    s = sentiment_features.vader_vector(comment)
    w = writing_style_features.writingstyle_vector(comment)
    l = lexicon_features.lexicon_vector(comment)
    return s+w+l


def embedding_sentiment_ws_lexicon_vector(comment):
    """
    Returns a Word-Embedding + VADER-Sentiment + Writing-Style + Lexicon vector for the given comment.

    Parameters
    ----------
    comment: str (comment text)

    Returns
    -------
    list of 310 features
        Order: [word_embedding] + [pos_score, neg_score, neu_score, word_count, url_count, pronoun_count, exclamation/question_count, capital_word_count, hate_count, non_eng_count]
    """
    embed = embedding_features.calculate_vector(model=GLOVE_VEC, text=comment)
    s = sentiment_features.vader_vector(comment)
    w = writing_style_features.writingstyle_vector(comment)
    l = lexicon_features.lexicon_vector(comment)
    return list(embed)+s+w+l


def feature_vectorize(main_sentences, context_sentences, use_comments, function_call):
    """
    Vectorizes the given list of comments using the given function_call. The function_call determines which combinations of features is to be used. Can choose to use Comment_Text (will be concatenate at the end).

    Parameters
    ----------
    main_sentences : list of reply sentences (or optionally just 'question' sentences)
    context_sentences : list of comment (context) sentences
    use_comments : boolean
        True if context comments are to be considered. Note that this should be False for 'Question Text Only' experiment.

    Returns
    -------
    numpy 2D array: vectorized input 'X'
    """
    X_reply = []
    for sent in main_sentences:
        X_reply.append(function_call(sent))
    X_reply = np.array(X_reply)

    if use_comments:
        print("Using Comment Text as well....")
        X_comment = []
        for sent in context_sentences:
            X_comment.append(function_call(sent))

        X_comment = np.array(X_comment)
        X = np.concatenate((X_reply, X_comment), axis=1)

    else:
        X = X_reply
    return X

GLOVE_VEC = embedding_features.loadGloveModel('/path/word_embeddings/glove.840B.300d.txt')
