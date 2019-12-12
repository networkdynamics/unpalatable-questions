import os
import sys
import argparse
import pickle
from run import run_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Name of model to be run', required=True)
parser.add_argument('--elmo', help='Use ELMo embeddings', action="store_true")
parser.add_argument('--save_preds', help='Save predictions for Error Analysis', action="store_true")
parser.add_argument('--save_model', help='Save model weights & vocabulary', action="store_true")
args = parser.parse_args()

error_path = '/home/ndg/users/sbagga1/unpalatable-questions/pickles/error-analysis/'
results_path = '/home/ndg/users/sbagga1/unpalatable-questions/results/model-results/'+args.model+'_'+str(args.elmo)+'.tsv'
if os.path.exists(results_path):
    sys.exit("Results file already exists: " + results_path)
results_file = open(results_path, "w")
results_file.write("Agreement\tContext\tModel\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\n")

print("Model = {} | ELMo = {} | SavePreds = {} | SaveModel = {} | OutFname = {}".format(args.model, args.elmo, args.save_preds, args.save_model, results_path))

CONTEXTS = [('question', False), ('reply_text', False), ('reply_text', True)] # second value of tuple is for double_input | double_input = True means that preceding comment should be considered
AGREEMENTS = ['confidence-60', 'confidence-100']

for (context, double_input) in CONTEXTS:
    for conf in AGREEMENTS:
        f1, auroc, w_f1, precision, recall, accuracy, auprc, preds, n_epochs = run_model(args.model, context, conf,
                                                                                         double_input=double_input,
                                                                                         use_elmo=args.elmo,
                                                                                         save_predictions=args.save_preds,
                                                                                         save_model=args.save_model)
        results_file.write(str(float(conf.split('-')[1])/100)+'\t'+context+'_'+str(double_input)+'\t'+args.model+'_'+str(args.elmo)+'_'+str(n_epochs)+'\t'+str(f1)+'\t'+str(auroc)+'\t'+str(w_f1)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')
        if args.save_preds:
            with open(error_path+args.model+str(args.elmo)+'_'+context+'_'+str(double_input)+'_conf_'+conf.split('-')[1]+'.pickle', 'wb') as f:
                pickle.dump(preds, f)
