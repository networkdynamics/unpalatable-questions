"""
This file has functions to vectorize based on n-grams only (word, char, multiple ranges) and n-grams combined with the other feature categories.
"""
import feature_combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack
from scipy import sparse

def ngrams_vectorize(question_train_sentences, question_test_sentences, reply_train_sentences, reply_test_sentences,
                     comment_train_sentences, comment_test_sentences, ngram_range, analyzer, use_comments, use_question_only=False):
    """
    Vectorizes the input text using bag of n-grams approach. Functionality to only consider the reply when vectorizing
    or also concatenate the context comment vectors. If the comment sentences are empty lists, they are not concatenated.

    Parameters
    ----------
    question_train_sentences: numpy array of question sentences from the training dataset
    question_test_sentences: numpy array of question sentences from the testing dataset
    reply_train_sentences: numpy array of reply sentences from the training dataset
    reply_test_sentences: numpy array of reply sentences from the testing dataset
    comment_train_sentences: numpy array of comment (context) sentences from the training dataset
    comment_test_sentences: numpy array of comment (context) sentences from the testing dataset
    ngram_range: tuple for the value of n in n-grams
    analyzer: str - 'word' for word n-grams; 'char' for character n-grams
    use_comments: boolean - True if we want to use the Comment Text. Can not be True if use_question_only is True.
    use_question_only: boolean - True if we want to use the Question Text ONLY; False otherwise

    Returns
    -------
    X_train, X_test
    """
    if use_question_only:
#         print("Using Question Text only....")
        count_vec_ques = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
        ques_train_counts = count_vec_ques.fit_transform(question_train_sentences)
        ques_test_counts = count_vec_ques.transform(question_test_sentences)
        return ques_train_counts, ques_test_counts

    # "Reply_Text" will always be used if use_question_only is False:
    count_vec_reply = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    reply_train_counts = count_vec_reply.fit_transform(reply_train_sentences)
    reply_test_counts = count_vec_reply.transform(reply_test_sentences)

    if use_comments: # concatenate the vectors (add sparse matrices)
#         print("Using both Reply Text and Comment Text....")
        count_vec_comment = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
        comment_train_counts = count_vec_comment.fit_transform(comment_train_sentences)
        comment_test_counts = count_vec_comment.transform(comment_test_sentences)
        # Combining context: adding the sparse matrices column-wise and converting the coo matrix to csr
        X_train = hstack([reply_train_counts, comment_train_counts]).tocsr()
        X_test = hstack([reply_test_counts, comment_test_counts]).tocsr()

    else: # only considering "Reply_Text"
#         print("Using Reply Text only....")
        X_train = reply_train_counts
        X_test = reply_test_counts

    return X_train, X_test


def ngrams_combined_vectorize(question_train_sentences, question_test_sentences, reply_train_sentences, reply_test_sentences,
                     comment_train_sentences, comment_test_sentences, ngram_range, analyzer, use_comments, use_question_only=False):
    """
    Vectorizes the input text using bag of n-grams approach + other features (writing-style + lexicon + sentiment)
    Functionality to only consider the reply when vectorizing or also concatenate the context comment vectors.
    If the comment sentences are empty lists, they are not concatenated.

    Parameters
    ----------
    question_train_sentences: numpy array of question sentences from the training dataset
    question_test_sentences: numpy array of question sentences from the testing dataset
    reply_train_sentences: numpy array of reply sentences from the training dataset
    reply_test_sentences: numpy array of reply sentences from the testing dataset
    comment_train_sentences: numpy array of comment (context) sentences from the training dataset
    comment_test_sentences: numpy array of comment (context) sentences from the testing dataset
    ngram_range: tuple for the value of n in n-grams
    analyzer: str - 'word' for word n-grams; 'char' for character n-grams
    use_comments: boolean - True if we want to use the Comment Text. Can not be True if use_question_only is True.
    use_question_only: boolean - True if we want to use the Question Text ONLY; False otherwise

    Returns
    -------
    X_train, X_test
    """
    X_extra_train = []; X_extra_test = []

    if use_question_only:
#         print("Using Question Text only....")
        count_vec_ques = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
        ques_train_counts = count_vec_ques.fit_transform(question_train_sentences)
        ques_test_counts = count_vec_ques.transform(question_test_sentences)
#         print("BEFORE: ", ques_train_counts.shape, ques_test_counts.shape)

        for text in question_train_sentences:
            vec = feature_combinations.sentiment_ws_lexicon_vector(text)
            X_extra_train.append(vec)

        for text in question_test_sentences:
            vec = feature_combinations.sentiment_ws_lexicon_vector(text)
            X_extra_test.append(vec)

        X_extra_train = sparse.csr_matrix(X_extra_train); X_extra_test = sparse.csr_matrix(X_extra_test)
        X_train = sparse.hstack([ques_train_counts, X_extra_train]).tocsr()
        X_test = sparse.hstack([ques_test_counts, X_extra_test]).tocsr()

#         print("AFTER: ", X_train.shape, X_test.shape)
        return X_train, X_test

    # "Reply_Text" will always be used if use_question_only is False:
    count_vec_reply = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    reply_train_counts = count_vec_reply.fit_transform(reply_train_sentences)
    reply_test_counts = count_vec_reply.transform(reply_test_sentences)

    if use_comments: # concatenate the vectors (add sparse matrices)
#         print("Using both Reply Text and Comment Text....")
        count_vec_comment = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
        comment_train_counts = count_vec_comment.fit_transform(comment_train_sentences)
        comment_test_counts = count_vec_comment.transform(comment_test_sentences)
        # Combining context (within n-grams): adding the sparse matrices column-wise and converting the coo matrix to csr
        X_train_ngrams = hstack([reply_train_counts, comment_train_counts]).tocsr()
        X_test_ngrams = hstack([reply_test_counts, comment_test_counts]).tocsr()
#         print("BEFORE: ", X_train_ngrams.shape, X_test_ngrams.shape)

        # Sanity check:
        if len(reply_train_sentences) != len(comment_train_sentences):
            sys.exit("ERROR: Reply and Comments don't correspond in length (train).")
        if len(reply_test_sentences) != len(comment_test_sentences):
            sys.exit("ERROR: Reply and Comments don't correspond in length (test).")

        # Combining context (for other features)
        for i, _ in enumerate(reply_train_sentences):
            reply = reply_train_sentences[i]
            comment = comment_train_sentences[i]
            vec1 = feature_combinations.sentiment_ws_lexicon_vector(reply)
            vec2 = feature_combinations.sentiment_ws_lexicon_vector(comment)
            X_extra_train.append(vec1+vec2)

        for i, _ in enumerate(reply_test_sentences):
            reply = reply_test_sentences[i]
            comment = comment_test_sentences[i]
            vec1 = feature_combinations.sentiment_ws_lexicon_vector(reply)
            vec2 = feature_combinations.sentiment_ws_lexicon_vector(comment)
            X_extra_test.append(vec1+vec2)


    else: # only considering "Reply_Text"
#         print("Using Reply Text only....")
        X_train_ngrams = reply_train_counts
        X_test_ngrams = reply_test_counts
#         print("BEFORE: ", X_train_ngrams.shape, X_test_ngrams.shape)
        for text in reply_train_sentences:
            vec = feature_combinations.sentiment_ws_lexicon_vector(text)
            X_extra_train.append(vec)

        for text in reply_test_sentences:
            vec = feature_combinations.sentiment_ws_lexicon_vector(text)
            X_extra_test.append(vec)

    X_extra_train = sparse.csr_matrix(X_extra_train)
    X_extra_test = sparse.csr_matrix(X_extra_test)

    X_train = sparse.hstack([X_train_ngrams, X_extra_train]).tocsr()
    X_test = sparse.hstack([X_test_ngrams, X_extra_test]).tocsr()

#     print("AFTER: ", X_train.shape, X_test.shape)
    return X_train, X_test
