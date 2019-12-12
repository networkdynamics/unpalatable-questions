import numpy as np
import pandas as pd

def remove_nonASCII(comment):
    return ''.join([i if ord(i) < 128 else ' ' for i in comment])


def load_dataset(path, conf=0.6, return_ids=False):
    """
    Loads the annotated dataset that have confidence/agreement >= 'conf'. Possible conf values = 0.6 | 0.8 | 1.0
    
    'yes_unpalatable'=1 | 'not_unpalatable'=0
    
    If return_ids is True, also returns a corresponding list of unique IDs for each row.
    """
    if conf not in [0.6, 0.8, 1.0]:
        print("Error: Confidence must be either 0.6, 0.8, or 1.0")
        return None
    
    df = pd.read_csv(path, lineterminator='\n')
    
    # Sanity check: make sure the line_terminator of Pandas doesn't mess things up
    fname = path.split('/')[-1]
    number_of_rows = int(fname.split('.')[0].split('_')[-1]) # no. of rows there should be
    if df.shape[0] != number_of_rows:
        sys.exit("lineterminator Error: CSV not read properly")
    else:
        print("{} rows from {}\nDataFrame was read correctly!".format(df.shape[0], fname))
    
    
    
    df = df.loc[df['confidence']>=conf] # filter by confidence:
    print("{} has {} rows with confidence >= {}".format(path, df.shape[0], conf))

    questions, replies, comments, y, IDs = [], [], [], [], []
    for i in df['question'].tolist():
        questions.append(remove_nonASCII(i))
    for i in df['reply_text'].tolist():
        replies.append(remove_nonASCII(i))
    for i in df['comment_text'].tolist():
        comments.append(remove_nonASCII(i))
    for i in df['label'].tolist():
        if i == 'yes_unpalatable':
            y.append(1)
        elif i == 'not_unpalatable':
            y.append(0)
        else:
            print("LabelError: ", i)
    for i in df['reply_id'].tolist():
        IDs.append(i)

    questions = np.array(questions); replies = np.array(replies); comments = np.array(comments); y = np.array(y); IDs = np.array(IDs)
    if return_ids:
        print("Returning unique reply IDs as well..")
        return questions, replies, comments, y, IDs        
    else:
        return questions, replies, comments, y