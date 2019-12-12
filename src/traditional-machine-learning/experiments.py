"""
- Runs different feature categories (individually and their combinations). Includes multiple word embedding models (pre-trained word2vec, GloVe, Fasttext; and Reddit-trained word2vec).
- Saves the best f1-score along with the AUROC, Weighted f1 etc. to a TSV.
- Three modeling cases: only Question Text, full Reply Text, full Reply Text + preceding Comment Text
- Covers different confidence agreement thresholds in the annotated dataset.
- 5-fold cross validated hyperparamter tuning for the machine learning algorithms.
See experiments_ngram.py for experiments involving n-gram features.
"""
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import loader
import embedding_features
import writing_style_features
import feature_combinations


def hyperparm_tuning(X1):
    """
    Runs the machine learning algorithm (SVM or others) with different hyperparameters; optimises for the best F1;
    Scales the feature vector.
    Also reports the corresponding AUROC and Weighted F1.

    Parameters
    ----------
    X: numpy 2D array
        X is the vectorized input. Note that the corresponding labels 'y' are static.

    Returns
    -------
    dict: maps the three classification metrics to their values (for the best hyperparameters)
    """
    clf = GridSearchCV(algo, param_grid=tuned_parameters, cv=NUMBER_OF_FOLDS, scoring='f1') # Optimise for best F1
    X = scaler.fit_transform(X1)
    clf.fit(X, y)
    best_svm = clf.best_estimator_
#     print("\nBest f1-score achieved by SVM is {} | Corresponding hyperparameters: C={}; kernel={}.".format(clf.best_score_, best_svm.C, best_svm.kernel))

    output = {}
    # Run the best-f1 model to compute other corresponding metrics:
    for s in metrics:
        temp = np.array(cross_val_score(best_svm, X, y, cv=NUMBER_OF_FOLDS, scoring=s))
        print("Best {} = {}".format(s, temp.mean()))
        output[s] = temp.mean()
    return output


def embedding_experiments(use_comments, use_question_only):
    """
    4 word embeddings: pre-trained word2vec, fasttext, GloVe | and Reddit-trained word2vec (all of 2016 data).
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    for model_name, embed_model in models:
        print("\n\n--------------------------\nWord-Embeddings | Model = {} | Using Comments = {} | Using Question Only = {}".format(model_name, use_comments, use_question_only))

        if use_question_only: # only Question Text
            if use_comments: # sanity check
                sys.exit("use_question_only and use_comments can not be True at the same time")

            X = embedding_features.embedding_vectorize(model=embed_model, main_sentences=question_sentences,
                                                       context_sentences=[], use_comments=False)
            f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

        else: # Either only Reply Text or both Reply+Comment Text
            X = embedding_features.embedding_vectorize(model=embed_model, main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments)
            if use_comments:
                f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
            else:
                f.write(str(CONFIDENCE)+'\tReply Text\t')

        score_dict = hyperparm_tuning(X)
        f.write(model_name+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
                +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')


def writing_style_experiments(use_comments, use_question_only):
    """
    Feature vector includes writing style features: word count, pronoun_count, url_count, exclamation/question_mark_count, uppercase_count
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    print("\n\n--------------------------\n Writing-Style | Using Comments = {} | Using Question Only = {}".format(use_comments, use_question_only))

    if use_question_only: # only Question Text
        if use_comments: # sanity check
            sys.exit("use_question_only and use_comments can not be True at the same time")

        X = writing_style_features.style_vectorize(main_sentences=question_sentences, context_sentences=[], use_comments=False)
        f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

    else: # Either only Reply Text or both Reply+Comment Text
        X = writing_style_features.style_vectorize(main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments)
        if use_comments:
            f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
        else:
            f.write(str(CONFIDENCE)+'\tReply Text\t')

    score_dict = hyperparm_tuning(X)
    f.write('Writing Style'+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
            +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')

def style_lexicon_experiments(use_comments, use_question_only):
    """
    Feature vector includes writing style features + lexicon features: word count, pronoun_count, url_count, exclamation/question_mark_count, uppercase_count
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    print("\n\n--------------------------\n Writing-Style + Lexicon | Using Comments = {} | Using Question Only = {}".format(use_comments, use_question_only))

    if use_question_only: # only Question Text
        if use_comments: # sanity check
            sys.exit("use_question_only and use_comments can not be True at the same time")

        X = feature_combinations.feature_vectorize(main_sentences=question_sentences, context_sentences=[], use_comments=False, function_call=feature_combinations.ws_lexicon_vector)
        f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

    else: # Either only Reply Text or both Reply+Comment Text
        X = feature_combinations.feature_vectorize(main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments, function_call=feature_combinations.ws_lexicon_vector)
        if use_comments:
            f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
        else:
            f.write(str(CONFIDENCE)+'\tReply Text\t')

    score_dict = hyperparm_tuning(X)
    f.write('Writing Style + Lexicon'+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
            +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')


def embedding_ws_lexicon_experiments(use_comments, use_question_only):
    """
    Feature vector includes Reddit-trained word embeddings, writing style features, lexicon features.
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    print("\n\n--------------------------\nWord-Embeddings + Writing-Style + Lexicon | Using Comments = {} | Using Question Only = {}".format(use_comments, use_question_only))

    if use_question_only: # only Question Text
        if use_comments: # sanity check
            sys.exit("use_question_only and use_comments can not be True at the same time")

        X = feature_combinations.feature_vectorize(main_sentences=question_sentences, context_sentences=[], use_comments=False, function_call=feature_combinations.embedding_ws_lexicon_vector)
        f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

    else: # Either only Reply Text or both Reply+Comment Text
        X = feature_combinations.feature_vectorize(main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments, function_call=feature_combinations.embedding_ws_lexicon_vector)
        if use_comments:
            f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
        else:
            f.write(str(CONFIDENCE)+'\tReply Text\t')

    score_dict = hyperparm_tuning(X)
    f.write('Writing Style + Lexicon + GloVe'+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
            +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')


def sentiment_ws_lexicon_vector_experiments(use_comments, use_question_only):
    """
    Feature vector includes sentiment features, writing style features, lexicon features.
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    print("\n\n--------------------------\nSentiment + Writing-Style + Lexicon | Using Comments = {} | Using Question Only = {}".format(use_comments, use_question_only))

    if use_question_only: # only Question Text
        if use_comments: # sanity check
            sys.exit("use_question_only and use_comments can not be True at the same time")

        X = feature_combinations.feature_vectorize(main_sentences=question_sentences, context_sentences=[], use_comments=False, function_call=feature_combinations.sentiment_ws_lexicon_vector)
        f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

    else: # Either only Reply Text or both Reply+Comment Text
        X = feature_combinations.feature_vectorize(main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments, function_call=feature_combinations.sentiment_ws_lexicon_vector)
        if use_comments:
            f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
        else:
            f.write(str(CONFIDENCE)+'\tReply Text\t')

    score_dict = hyperparm_tuning(X)
    f.write('Writing Style + Lexicon + Sentiment'+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
            +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')


def embedding_sentiment_ws_lexicon_vector_experiments(use_comments, use_question_only):
    """
    Feature vector includes Reddit-trained word embedding, sentiment features, writing style features, lexicon features.
    Prints the best f1 (corresponding AUC, weighted f1) achieved by the machine learnign algorithm with hyperparameter tuning.

    Parameters
    ----------
    use_comments: boolean
        Consider context comments if True (must be False if use_question_only is True)
    use_question_only: booolean
        Consider only the Question Text if True
    """
    print("\n\n--------------------------\nWord-Embedding + Sentiment + Writing-Style + Lexicon | Using Comments = {} | Using Question Only = {}".format(use_comments, use_question_only))

    if use_question_only: # only Question Text
        if use_comments: # sanity check
            sys.exit("use_question_only and use_comments can not be True at the same time")

        X = feature_combinations.feature_vectorize(main_sentences=question_sentences, context_sentences=[], use_comments=False, function_call=feature_combinations.embedding_sentiment_ws_lexicon_vector)
        f.write(str(CONFIDENCE)+'\tQuestion Text Only\t')

    else: # Either only Reply Text or both Reply+Comment Text
        X = feature_combinations.feature_vectorize(main_sentences=reply_sentences, context_sentences=comment_sentences, use_comments=use_comments, function_call=feature_combinations.embedding_sentiment_ws_lexicon_vector)
        if use_comments:
            f.write(str(CONFIDENCE)+'\tReply Text + Comment Text\t')
        else:
            f.write(str(CONFIDENCE)+'\tReply Text\t')

    score_dict = hyperparm_tuning(X)
    f.write('Writing Style + Lexicon + Sentiment + GloVe'+'\t'+str(score_dict['f1'])+'\t'+str(score_dict['roc_auc'])+'\t'+str(score_dict['f1_weighted'])
            +'\t'+str(score_dict['precision'])+'\t'+str(score_dict['recall'])+'\t'+str(score_dict['accuracy'])+'\t'+str(score_dict['average_precision'])+'\n')


if __name__ == '__main__':
    # Note: SVM took way too much time to run and results were comparable (often worse)
#     algo = SVC(probability=True)
#     tuned_parameters = [{'C': [1, 100, 10000], 'kernel': ['linear']}, {'C': [1, 100, 10000], 'gamma': ['auto', 0.001, 0.0001], 'kernel': ['rbf']}]

    NUMBER_OF_FOLDS = 5

    algo = LogisticRegression()
    tuned_parameters = [{'C': [0.001, 1, 1000], 'penalty': ['l1'], 'solver': ['liblinear']},
                        {'C': [0.001, 1, 1000], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs']}]

    scaler = StandardScaler()
    metrics = ['f1', 'roc_auc', 'f1_weighted', 'precision', 'recall', 'accuracy', 'average_precision']

    # Load the models for embedding experiments:
    google_vec, fast_text_vec, glove_vec, reddit_vec = embedding_features.load_embedding_models()
    models = [('GoogleW2V', google_vec), ('Fasttext', fast_text_vec), ('GloVe', glove_vec), ('RedditW2V', reddit_vec)]

    data_path = '/data/annotations_UQ.csv'

    results_path = '/path/unpalatable-questions/results/model-results/LogReg_results_experiments.tsv'
    f = open(results_path, "w")
    f.write("Agreement\tContext\tModel\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\n")

    for CONFIDENCE in [0.6, 1.0]: # 3/5 and 5/5 agreement
        question_sentences, reply_sentences, comment_sentences, y = loader.load_dataset(data_path, conf=CONFIDENCE)
        # Only Question Text:
        embedding_experiments(use_comments=False, use_question_only=True)
        writing_style_experiments(use_comments=False, use_question_only=True)
        style_lexicon_experiments(use_comments=False, use_question_only=True)
        embedding_ws_lexicon_experiments(use_comments=False, use_question_only=True)
        sentiment_ws_lexicon_vector_experiments(use_comments=False, use_question_only=True)
        embedding_sentiment_ws_lexicon_vector_experiments(use_comments=False, use_question_only=True)
        for b in [True, False]:
            embedding_experiments(use_comments=b, use_question_only=False)
            writing_style_experiments(use_comments=b, use_question_only=False)
            style_lexicon_experiments(use_comments=b, use_question_only=False)
            embedding_ws_lexicon_experiments(use_comments=b, use_question_only=False)
            sentiment_ws_lexicon_vector_experiments(use_comments=b, use_question_only=False)
            embedding_sentiment_ws_lexicon_vector_experiments(use_comments=b, use_question_only=False)
    f.close()
