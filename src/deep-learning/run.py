"""
Author: Sunyam

This file runs various deep learning models for text classification.
"""
import data_reader
import train
import evaluate

import torch
import os
import sys
import pickle
import numpy as np

from pathlib import Path
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer # for ELMo
from allennlp.data.token_indexers import PretrainedBertIndexer # for BERT

def tokenizer(x: str):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

def tokenizer_bert(s: str):
    return bert_token_indexer.wordpiece_tokenizer(s)

def run_model(name, context, conf, double_input, use_elmo=False, save_predictions=False, save_model=False):
    """
    Runs the given model 'name' for the given 'context' and agreement level 'conf'. If double_input is True, runs the combined model using context comment text. Optionally saves the trained model & its vocabulary, and predictions.
    Allowed names: lstm | bilstm | stacked_bilstm | cnn | dense_lstm | dense_bilstm | dense_stacked_bilstm | dense_cnn | nli_cnn | bert | dense_bert
    
    If use_elmo=True, uses ELMo's pre-trained language model for embeddings.

    """
    if use_elmo:
        token_indexer = ELMoTokenCharactersIndexer() # token indexer is responsible for mapping tokens to integers: this makes sure that the mapping is consistent with what was used in the original ELMo training.
    elif name == 'bert':
        global bert_token_indexer
        bert_token_indexer = PretrainedBertIndexer(pretrained_model=BERT_MODEL, do_lowercase=True)
    else:
        token_indexer = SingleIdTokenIndexer()

    if name == 'bert': # BERT uses a special wordpiece tokenizer
        reader = data_reader.UnpalatableDatasetReader(main_input=context, additional_context=double_input,
                                                      tokenizer=tokenizer_bert, token_indexers={"tokens": bert_token_indexer},
                                                      label_cols=LABEL_COLS)
    else:
        reader = data_reader.UnpalatableDatasetReader(main_input=context, additional_context=double_input, tokenizer=tokenizer,
                                                      token_indexers={"tokens": token_indexer}, label_cols=LABEL_COLS)


    map_reply_id_pred_probability = {}; n_epochs = []
    f1s, AUROCs, weighted_f1s, precision_s, recall_s, accuracies, AUPRCs = [], [], [], [], [], [], []

    for fold_number in range(1,6): # 5-fold cross validation
        train_fname = 'train_data_fold_'+str(fold_number)+'_OneHot.csv'
        val_fname = 'val_data_fold_'+str(fold_number)+'_OneHot.csv'
        test_fname = 'test_data_fold_'+str(fold_number)+'_OneHot.csv'

        train_dataset = reader.read(file_path=DATA_ROOT / conf / train_fname)
        validation_dataset = reader.read(file_path=DATA_ROOT / conf / val_fname)
        test_dataset = reader.read(file_path=DATA_ROOT / conf / test_fname)
        print("\n#####################################################\n", double_input, context, conf, name, len(train_dataset), len(validation_dataset), len(test_dataset))

        # Train model:
        if name == 'lstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=False,
                                                num_layers=1, bidirectional=False, use_elmo=use_elmo, double_input=double_input)
        elif name == 'dense_lstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=True,
                                                col_name=context, num_layers=1, bidirectional=False, use_elmo=use_elmo,
                                                double_input=double_input)
        elif name == 'bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=False,
                                                num_layers=1, bidirectional=True, use_elmo=use_elmo, double_input=double_input)
        elif name == 'dense_bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=True,
                                                col_name=context, num_layers=1, bidirectional=True, use_elmo=use_elmo,
                                                double_input=double_input)
        elif name == 'stacked_bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=False,
                                                num_layers=2, bidirectional=True, use_elmo=use_elmo, double_input=double_input)
        elif name == 'dense_stacked_bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=True,
                                                col_name=context, num_layers=2, bidirectional=True, use_elmo=use_elmo,
                                                double_input=double_input)
        elif name == 'cnn':
            if context == 'reply_text': filter_sizes = (2,3) # kernels can not be bigger than the shortest sentence
            else: filter_sizes = (2,)
            model, vocab, ep = train.train_cnn(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=False,
                                               num_filters=100, filter_sizes=filter_sizes, use_elmo=use_elmo,
                                               double_input=double_input)
        elif name == 'dense_cnn':
            if context == 'reply_text': filter_sizes = (2,3) # kernels can not be bigger than the shortest sentence
            else: filter_sizes = (2,)
            model, vocab, ep = train.train_cnn(train_dataset, validation_dataset, BATCH_SIZE, dense_vector=True,
                                               col_name=context, num_filters=100, filter_sizes=filter_sizes, use_elmo=use_elmo,
                                               double_input=double_input)      
        elif name == 'nli_cnn':
            if double_input == False:
                print("Error: NLI-inspired architecture only accepts double-input.")
                return [None]*9
            filter_sizes = (2,3)
            model, vocab, ep = train.train_nli(train_dataset, validation_dataset, BATCH_SIZE, use_elmo=use_elmo,
                                               num_filters=100, filter_sizes=filter_sizes)
        elif name == 'bert':
            model, vocab, ep = train.train_bert(train_dataset, validation_dataset, BATCH_SIZE, pretrained_model=BERT_MODEL,
                                                dense_vector=False, double_input=double_input)
        elif name == 'dense_bert':
            model, vocab, ep = train.train_bert(train_dataset, validation_dataset, BATCH_SIZE, pretrained_model=BERT_MODEL, 
                                                dense_vector=True, col_name=context, double_input=double_input)
        else:
            sys.exit("'name' not valid")
            
        n_epochs.append(ep) # keep track of number of actual training epochs for each fold
        
        # Predict and evaluate model on test set:
        preds = evaluate.make_predictions(model, vocab, test_dataset, BATCH_SIZE, use_gpu=False) # NOTE: preds is of shape (number of samples, 2) - the columns represent the probabilities for the two classes in order ['yes_unp', 'not_unp']
        f1, auroc, w_f1, precision, recall, acc, auprc = evaluate.compute_metrics(preds, test_dataset)
        
        if save_predictions: # save predictions for error analysis
            replyid_pred = evaluate.map_id_prediction(preds, test_dataset)
            if set(replyid_pred.keys()).intersection(set(map_reply_id_pred_probability.keys())) != set(): # sanity check
                sys.exit("Error: There is overlap in Test IDs across folds.")
            map_reply_id_pred_probability.update(replyid_pred)
        
        if save_model: # save the model weights and vocabulary
            with open('./tmp/'+name+'_model_conf_'+conf.split('-')[1]+'_fold_'+str(fold_number)+'.th', 'wb') as f:
                torch.save(model.state_dict(), f)
            vocab.save_to_files("./tmp/"+name+"_vocabulary_"+conf.split('-')[1]+"_fold_"+str(fold_number))

        print("\nFold #{} | F1 = {} | AUROC = {} | AUPRC = {}".format(fold_number, f1, auroc, auprc))

        f1s.append(f1); AUROCs.append(auroc); weighted_f1s.append(w_f1); precision_s.append(precision); 
        recall_s.append(recall); accuracies.append(acc); AUPRCs.append(auprc)

    mean_f1 = np.array(f1s).mean(); mean_auroc = np.array(AUROCs).mean(); mean_weighted_f1 = np.array(weighted_f1s).mean(); 
    mean_precision = np.array(precision_s).mean(); mean_recall = np.array(recall_s).mean(); mean_accuracy = np.array(accuracies).mean(); mean_auprc = np.array(AUPRCs).mean()

    print("Total predictions: {} | Save Predictions: {}".format(len(map_reply_id_pred_probability), save_predictions))
    
    return mean_f1, mean_auroc, mean_weighted_f1, mean_precision, mean_recall, mean_accuracy, mean_auprc, map_reply_id_pred_probability, n_epochs

BATCH_SIZE = 64
DATA_ROOT = Path("/home/ndg/users/sbagga1/unpalatable-questions/data-folds/")
LABEL_COLS = ["yes_unpalatable", "not_unpalatable"] # Ordering is important: ['yes_unp', 'not_unp']
BERT_MODEL = "bert-large-uncased"
torch.manual_seed(42)