"""
- Runs multiple word n-gram and character n-gram experiments for different values of n; also combined with other feature categories.
- Reports the best f1-score along with the AUROC, Weighted f1, and other metrics.
- Three modeling cases: only Question Text, full Reply Text, full Reply Text + preceding Comment Text
- Covers different confidence agreement thresholds in the annotated dataset.
- 5-fold cross validated hyperparamter tuning for LogReg/SVM.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
import loader
import ngram_features


def hyperparameter_tuning(ngram_range, analyzer, use_question_only, use_comments, vectorize_fn):
    """
    Runs the given algorithm with hyperparameter tuning (5-fold CV) for the given ngram_range and other conditions. Optimises for the best f1. The method to vectorize the text is also passed as input.

    Parameters
    ----------
    ngram_range: tuple for the value of n
    analyzer: str
        'word' for word n-grams and 'char' for character n-grams
    use_question_only: boolean
        True if only Question Text is to be considered
    use_comments: bool
        True if comment (context) sentences are to be considered
    vectorize_fn: Callable
        method to vectorize X (implemented in ngram_features.py)

    Returns the best f1; and corresponding: AUROC,weighted f1,precision,recall,accuracy,AUPRC.
    """
    best_f1 = 0.0

    for param_dict in param_object:
        # Set the desired hyperparameters:
        algo.set_params(**param_dict)

        f1s = []; AUROCs = []; weighted_f1s = []; precision_s = []; recall_s = []; accuracies = []; AUPRCs = []
        for train_indices, test_indices in skf.split(X=np.zeros(len(y)), y=y): # only really need 'y' for splitting

            X_train, X_test = vectorize_fn(question_train_sentences=question_sentences[train_indices],
                                           question_test_sentences=question_sentences[test_indices],
                                           reply_train_sentences=reply_sentences[train_indices],
                                           reply_test_sentences=reply_sentences[test_indices],
                                           comment_train_sentences=comment_sentences[train_indices],
                                           comment_test_sentences=comment_sentences[test_indices],
                                           ngram_range=ngram_range,
                                           analyzer=analyzer,
                                           use_comments=use_comments,
                                           use_question_only=use_question_only)

            y_train = y[train_indices]
            y_test = y[test_indices]

            # Scale the data: (Performance was worse for batch 1)
#             scaler = StandardScaler(with_mean=False)
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)

            clf = algo.fit(X_train, y_train)
            preds = clf.predict(X_test)
            preds_with_probs = clf.predict_proba(X_test) # For AUC

            # Compute classification metrics:
            f1 = f1_score(y_test, preds)
            auroc = roc_auc_score(y_test, preds_with_probs[:,1]) # need to pass probabilities for AUROC
            w_f1 = f1_score(y_test, preds, average='weighted')
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            auprc = average_precision_score(y_test, preds_with_probs[:,1]) # Area under Precision-Recall curve

            f1s.append(f1)
            AUROCs.append(auroc)
            weighted_f1s.append(w_f1)
            precision_s.append(precision)
            recall_s.append(recall)
            accuracies.append(acc)
            AUPRCs.append(auprc)

        f1s = np.array(f1s); AUROCs = np.array(AUROCs); weighted_f1s = np.array(weighted_f1s); precision_s = np.array(precision_s); recall_s = np.array(recall_s); accuracies = np.array(accuracies); AUPRCs = np.array(AUPRCs)
        mean_f1 = f1s.mean(); mean_auroc = AUROCs.mean(); mean_weighted_f1 = weighted_f1s.mean(); mean_precision = precision_s.mean(); mean_recall = recall_s.mean(); mean_accuracy = accuracies.mean(); mean_auprc = AUPRCs.mean()

        if mean_f1 > best_f1: # Keep track of best f1 and corresponding metrics
            best_f1 = mean_f1
            corresponding_auroc = mean_auroc
            corresponding_weighted_f1 = mean_weighted_f1
            corresponding_precision = mean_precision
            corresponding_recall = mean_recall
            corresponding_accuracy = mean_accuracy
            corresponding_auprc = mean_auprc

    return best_f1, corresponding_auroc, corresponding_weighted_f1, corresponding_precision, corresponding_recall, corresponding_accuracy, corresponding_auprc



def random_baseline(y):
    """
    Random baseline considreing equal probability distribution and the actual distribtuion in annotated data.

    Parameters
    ----------
    y: list of labels (1s and 0s)
    """
    f1s = []; weighted_f1s = []; aucs = []; auprcs = []; precs = []; recalls = []

    n_sims = 1000 # number of simulations

    p = [[0.5, 0.5], [0.18, 0.82]] # Order is 1 (Unpalatable), 0 (NotUnpalatable)

    for prob in p:
        for _ in range(n_sims):
            preds = []
            for i in range(len(y)):
                random_prediction = np.random.choice([1,0], p=prob)
                preds.append(random_prediction)
            f1 = f1_score(y, preds)
            prec = precision_score(y, preds)
            recall = recall_score(y, preds)
            f1_w = f1_score(y, preds, average='weighted')
            auc = roc_auc_score(y, preds)
            auprc = average_precision_score(y, preds)
            f1s.append(f1)
            weighted_f1s.append(f1_w)
            aucs.append(auc)
            auprcs.append(auprc)
            precs.append(prec)
            recalls.append(recall)
        print("\nRandom Baseline with distribution {}:".format(prob))
        print("F1: {} | AUROC: {} | Weighted F1: {} | AUPRC: {} | Precision: {} | Recall: {}".format(np.array(f1s).mean(), np.array(aucs).mean(), np.array(weighted_f1s).mean(), np.array(auprcs).mean(), np.array(precs).mean(), np.array(recalls).mean()))


def run_experiments(analyzer):
    """
    Runs experiments for different n-gram ranges and the given analyzer.

    Parameters
    ----------
    analyzer: str
        'word' to run word n-grams and 'char' to run char n-grams
    """
    for ngram_range in map_analyzer_ngram_options[analyzer]:
        ## Question Text Only ##
        print("\nRunning {} ngram-range = {} | Using only Question Text".format(analyzer, ngram_range))
        f1, auc, weighted_f1, prec, rec, accuracy, auprc = hyperparameter_tuning(ngram_range, analyzer=analyzer, use_question_only=True, use_comments=False, vectorize_fn=ngram_features.ngrams_vectorize)
        results_file.write(str(conf)+'\tQuestion Text Only\t'+analyzer+' '+str(ngram_range)+'\t'+str(f1)+'\t'+str(auc)+'\t'+str(weighted_f1)+'\t'+str(prec)+'\t'+str(rec)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')

        print("\nRunning {} ngram-range = {} combined with meta features (sentiment_ws_lexicon_vector) | Using only Question Text".format(analyzer, ngram_range))
        f1, auc, weighted_f1, prec, rec, accuracy, auprc = hyperparameter_tuning(ngram_range, analyzer=analyzer, use_question_only=True, use_comments=False, vectorize_fn=ngram_features.ngrams_combined_vectorize)
        results_file.write(str(conf)+'\tQuestion Text Only\t'+analyzer+' '+str(ngram_range)+' + Sentiment + Writing-Style + Lexicon\t'+str(f1)+'\t'+str(auc)+'\t'+str(weighted_f1)+'\t'+str(prec)+'\t'+str(rec)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')

        # N-grams only:
        for boolean in [True, False]:
            print("\nRunning {} ngram-range = {} | Using context comments = {}".format(analyzer, ngram_range, boolean))
            if boolean: # use_comments
                results_file.write(str(conf)+'\tReply Text + Comment Text\t'+analyzer+' '+str(ngram_range)+'\t')
            else:
                results_file.write(str(conf)+'\tReply Text\t'+analyzer+' '+str(ngram_range)+'\t')

            f1, auc, weighted_f1, prec, rec, accuracy, auprc = hyperparameter_tuning(ngram_range, analyzer=analyzer, use_question_only=False, use_comments=boolean, vectorize_fn=ngram_features.ngrams_vectorize)
            results_file.write(str(f1)+'\t'+str(auc)+'\t'+str(weighted_f1)+'\t'+str(prec)+'\t'+str(rec)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')

        # N-grams combined with Sentiment, Writing Style, Lexicon:
        for boolean in [True, False]:
            print("\nRunning {} ngram-range = {} combined with meta features (sentiment_ws_lexicon_vector) | Using context comments = {}".format(analyzer, ngram_range, boolean))
            if boolean: # use_comments
                results_file.write(str(conf)+'\tReply Text + Comment Text\t'+analyzer+' '+str(ngram_range)+' + Sentiment + Writing-Style + Lexicon\t')
            else:
                results_file.write(str(conf)+'\tReply Text\t'+analyzer+' '+str(ngram_range)+' + Sentiment + Writing-Style + Lexicon\t')

            f1, auc, weighted_f1, prec, rec, accuracy, auprc = hyperparameter_tuning(ngram_range, analyzer=analyzer, use_question_only=False, use_comments=boolean, vectorize_fn=ngram_features.ngrams_combined_vectorize)
            results_file.write(str(f1)+'\t'+str(auc)+'\t'+str(weighted_f1)+'\t'+str(prec)+'\t'+str(rec)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')


if __name__ == '__main__':
    # Note: SVM results were comparable (sometimes worse)
#     algo = SVC(probability=True)
#     tuned_parameters = [{'C': [1, 100, 10000], 'kernel': ['linear']}, {'C': [1, 100, 10000], 'gamma': ['auto', 0.001, 0.0001], 'kernel': ['rbf']}]

    NUMBER_OF_FOLDS = 5
    SEED = 42

    algo = LogisticRegression()
    tuned_parameters = [{'C': [0.001, 1, 1000], 'penalty': ['l1'], 'solver': ['liblinear']},
                        {'C': [0.001, 1, 1000], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs']}]

    param_object = ParameterGrid(tuned_parameters)
    skf = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, random_state=SEED) # Splits the data into 5 stratified folds

    # Word level: Unigrams, Bigrams, Trigrams, UniBiTri_combined | Character level: Trigrams, 4grams, 5grams, Tri45_combined
    map_analyzer_ngram_options = {'word': [(1,1), (2,2), (3,3), (1,3)], 'char': [(3,3), (4,4), (5,5), (3,5)]}

    data_path = '/data/annotations_UQ.csv'

    results_path = '/path/unpalatable-questions/results/model-results/LogReg_results.tsv'
    results_file = open(results_path, "w")
    results_file.write("Agreement\tContext\tModel\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\n")

    for conf in [0.6, 1.0]: # 3/5 and 5/5 agreement
        question_sentences, reply_sentences, comment_sentences, y = loader.load_dataset(data_path, conf=conf)
        random_baseline(y)
        for analyzer in ['word', 'char']:
            run_experiments(analyzer)
