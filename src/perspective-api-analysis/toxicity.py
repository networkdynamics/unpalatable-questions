"""
Author: Sunyam

The aim is to see how MTurk annotations compare with Perspective API's toxicity scores. This file saves the toxicity score returned by Perspective API for
(1) Question Text only, (2) Full Reply Text, and (3) Both Comment + Reply Text

The corresponding analysis is done in Analysis.ipynb
"""

import pandas as pd
import time
import json
from googleapiclient import discovery

def get_toxicity_score(text):
    """
    Calls the Perspective API to get the toxicity score for the given text.
    Thanks to: https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#python | Only works with Python 3.

    Parameters
    ----------
    text : str
        Comment text

    Returns
    -------
    float
        Perspective API's toxicity score
    """
    # Generates API client object dynamically based on service name and version:
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey='') # add key
    analyze_request = {
      'comment': {'text': text},
      'requestedAttributes': {'TOXICITY': {}}
    }
    response = service.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['summaryScore']['value']


def process_column(col_name):
    """
    Runs the given column of DataFrame through the Perspective API.
    Appends the toxicity score as an additional column to the DataFrame.
    """
    map_ID_text = dict([tuple(x) for x in df[['reply_id', col_name]].values])
    map_ID_toxicity_score = {}

    i = 0
    for ID, text in map_ID_text.items():
        try:
            map_ID_toxicity_score[ID] = get_toxicity_score(text)
        except Exception as e: # usually a 'language not supported' error
            print(e)
            print(text)

        time.sleep(2)
        i += 1
        if i % 1000 == 0:
            print("Done with {} samples for {}.".format(i, col_name))

    # Add to DataFrame:
    df['P_API_'+col_name] = df['reply_id'].map(map_ID_toxicity_score)
    print("\n\nDone with {} | Column added | DataFrame shape now: {}".format(col_name, df.shape))


if __name__ == '__main__':
    df = pd.read_csv('/data/annotations_UQ.csv', lineterminator='\n')
    df['all_text'] = df['comment_text'] + ' ' + df['reply_text'] # Combined Reply + Comment Text
    print("Initial DataFrame shape: ", df.shape)

    process_column(col_name='question')
    process_column(col_name='reply_text')
    process_column(col_name='all_text')

    df.to_csv('/home/ndg/users/sbagga1/unpalatable-questions/results/data_with_toxicity_score_MTurk.csv', line_terminator='\n', index=None)
