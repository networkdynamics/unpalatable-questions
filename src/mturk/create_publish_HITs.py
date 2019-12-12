"""
Author: Sunyam

This file creates and publishes Human Intelligence Tasks (HITs) to MTurk for Workers to complete.
"""
import boto3
import pandas as pd
import pickle
import os
import sys
import csv
import logging
import time
from random import shuffle
from collections import Counter


def create_qualification():
    """
    Creates a qualification type and returns the Qualification Type ID.
    """
    with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/XMLs/qual_test_questions.xml', 'r') as f:
        test_questions = f.read()

    with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/XMLs/qual_test_answers.xml', 'r') as f:
        answer_key = f.read()

    # Not specifying RetryDelayInSeconds which means that Workers can request this qualification only once
    qualification_type = mturk.create_qualification_type(
                                    Name='Test',
                                    Description='You will be presented with questions similar to the actual HITs. To qualify for our task, \
                                    please make sure you read the instructions very carefully before you begin the test.\
                                    Spending some time trying to understand the task\
                                    would be worth it as the instructions for all HITs are the same and you can do 100s of HITs if\
                                    you maintain high accuracy. Good luck!',
                                    QualificationTypeStatus='Active',
                                    Test=test_questions,
                                    AnswerKey=answer_key,
                                    TestDurationInSeconds=7200, # 120 minutes for entire qualification test
                                    AutoGranted=False
                                )

    qual_ID = qualification_type['QualificationType']['QualificationTypeId']
    logging.info("Qualification Type ID: %s", qual_ID)
    return qual_ID


def create_HIT_type(qualification_type_id):
    """
    Creates an HIT Type and returns the HIT Type ID.
    """
    hit_type = mturk.create_hit_type(
                        AutoApprovalDelayInSeconds=259200, # 3 days
                        AssignmentDurationInSeconds=3600, # 60 minutes (max time for one HIT)
                        Reward='0.5', # 50 cents for 10 comments
                        Title="Identify unpalatable questions (WARNING: This HIT may contain offensive language. Worker discretion is advised)",
                        Keywords='text, reddit, rude, abusive, impolite, question, socialmedia',
                        Description='You will be presented with Reddit comments that contain a question. Your task is to \
                        determine whether that question is unpalatable or rude to its recipient.',
                        QualificationRequirements=[
                            {
                                'QualificationTypeId': qualification_type_id,
                                'Comparator': 'GreaterThanOrEqualTo',
                                'IntegerValues': [10], # 100% accuracy on the test
                                'ActionsGuarded': 'Accept' # Worker cannot accept the HIT, but can preview and see the HIT in their search results
                            },
                        ]
                    )

    logging.info("HIT Type ID: %s\n\n", hit_type['HITTypeId'])
    return hit_type['HITTypeId']


def get_access_keys():
    """
    Reads the security credentials required to access the MTurk API. Recommended: use the keys are for an IAM user (not root credentials).
    """
    with open('/home/credentials.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header row
        for row in csv_reader:
            access_keys = row
    return access_keys


def get_published_questions():
    """
    Returns
    -------
    set
        reply IDs corresponding to questions that have already been published.
        NOTE: does not include any test questions; includes the 900 samples from Figure Eight run.
    """
    with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/published_reply_ids.txt', 'r') as f:
        IDs_PUBLISHED = f.read().splitlines()

    if len(IDs_PUBLISHED) != len(set(IDs_PUBLISHED)):
        logging.error("Duplicate reply IDs in published_reply_ids.txt;\
        Total count = {}; Unique count = {}".format(len(IDs_PUBLISHED), len(set(IDs_PUBLISHED))))

    IDs_PUBLISHED = set(IDs_PUBLISHED)
    logging.info("Number of unique samples already published: %s", len(IDs_PUBLISHED))
    return IDs_PUBLISHED


def get_published_test_questions():
    """
    Returns
    -------
    set
        reply IDs corresponding to already published Test Questions (including Qualification test questions)
    dict
        key = reply IDs | value = frequency of how many times they have been published.
    """
    with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/published_TQ_ids.txt', 'r') as f:
        TQs_USED = f.read().splitlines()

    if len(TQs_USED) != len(set(TQs_USED)): # NOTE: there will be duplicates because re-using Test Questions in batch 2 & 3
        logging.warning("Duplicate test questions in published_TQ_ids.txt;\
        Total count = {}; Unique count = {}".format(len(TQs_USED), len(set(TQs_USED))))

    logging.info("Number of unique TQs already published: %s", len(set(TQs_USED)))
    
    TQ_USED_DICT = dict(Counter(TQs_USED))
    
    return set(TQs_USED), TQ_USED_DICT


def extract_questions_for_single_HIT(N):
    """
    Selects N questions from the list of CSVs in /crowdsourcing/data/csvs.
    We do 10 questions in one HIT (9 questions + 1 secret TQ)

    Parameters
    ----------
    N: int
        number of actual questions per HIT (not including secret Test Question)

    Returns
    -------
    questions: list of N tuples
    tq: list of 1 tuple
        Ordering of all tuples is (question, reply, comment, comment_id, reply_id, subreddit)
    """
    path = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/'
    
    IDs_PUBLISHED = get_published_questions()
    TQs_USED, TQ_USED_DICT = get_published_test_questions()
    # There should be no overlap in published_reply_ids and published_TQ_ids: (there could be across batches if using high-confidence rows from batch 1 as Test Questions in batch 2)
    if IDs_PUBLISHED.intersection(TQs_USED) != set([]):
        logging.error('Overlap in published_reply_ids.txt and published_TQ_ids.txt!')
    
    ## Pick actual questions (non-TQ) ##
    questions = []
    for fname in os.listdir(path+'csvs/'):
        skipped = 0 # to keep track of how many rows get skipped
        df = pd.read_csv(path+'csvs/'+fname, lineterminator='\n')
        
        # Sanity check: make sure the line_terminator of Pandas doesn't mess things up
        number_of_rows = int(fname.split('.')[0].split('_')[-1]) # no. of rows there should be
        if df.shape[0] != number_of_rows:
            print "lineterminator Error: CSV not read properly"
            sys.exit("lineterminator Error: CSV not read properly")
        
        for row in df.values:
            question, reply, comment, comment_id, reply_id, subreddit = row
            
            if reply_id in IDs_PUBLISHED or reply_id in GOLD_TQs: # skip if already published or is a Test Question
                skipped += 1
                continue
                
            questions.append(tuple(row))
                
            if len(questions) == N: # break if you have the desired number of questions
                break        
        logging.info("{}: skipped {} reply IDs (total = {}) because they had already been published.".format(fname, skipped, df.shape[0]))
        
        if len(questions) == N: # break if you have the desired number of questions
            break        


    ## Pick a secret Test Question: pull one row from TestQuestions.csv ##
    tq_df = pd.read_csv(path+'TestQuestions.csv', lineterminator='\n')
    tq_df = tq_df.sample(frac=1) # Shuffle the rows
    
    if tq_df.columns.tolist() != ['question', 'reply_text', 'comment_text', 'comment_id', 'reply_id', 'subreddit', 'label']:
        sys.exit("Critical Error in column ordering of Test Question CSV")
    
    # Sample a new Test Question from the CSV (re-use a question if no new question found):
    new_TQ_flag = False
    for _, row in tq_df.iterrows():
        question, reply, comment, comment_id, reply_id, subreddit, label = row
        if reply_id not in TQs_USED: # new test question found
            tq = row
            new_TQ_flag = True
            break

    if new_TQ_flag == False: # If no new Test Question was found, randomly sample 1
        while True:
            tq = tq_df.sample(1).values.tolist()[0]
            question, reply, comment, comment_id, reply_id, subreddit, label = tq
            if TQ_USED_DICT[reply_id] >= 3: # look for another one if this has been used 3 or more times.
                continue
                print "{} TQ has been used {} times already. Skipping this one.".format(reply_id, TQ_USED_DICT[reply_id])
            else:
                break
                print "Found this {} TQ | has been used {} times. Awesome!".format(reply_id, TQ_USED_DICT[reply_id])

        logging.warning('Re-using a Test Question: %s', tq[4])

    return questions, list(tq[:-1]) # -1 to exclude 'label' column


def get_HIT_xml(N):
    """
    Modifies the question XML for a single HIT's layout. Contains multiple questions = N+1 (+1 for the secret Test Question).
    Also, appends the question reply IDs to /data/published_reply_ids.txt and appends the Test Question reply ID to /data/published_TQ_ids.txt

    Parameters
    ----------
    N: int
        number of actual questions per HIT (not including secret Test Question)
        
    Returns
    -------
    str
        Question XML with values inserted for the placeholder variables
    """
    path = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/'
    questions, tq = extract_questions_for_single_HIT(N)
    
    hit_questions = questions + [tuple(tq)]
    shuffle(hit_questions) # To ensure random ordering for the secret test question

    # Sanity check:
    if len(hit_questions) != N+1:
        logging.critical('Incorrect number of questions in a single HIT: %s (including secret TQ)', len(hit_questions))
        sys.exit('Error: Incorrect number of questions in a single HIT!')
        
    # Append the reply IDs:
    with open(path+'data/published_reply_ids.txt', 'a') as f:
        for row in questions:
            question, reply, comment, comment_id, reply_id, subreddit = row
            f.write("%s\n" % reply_id)
    logging.info("Added {} IDs to published_reply_ids.txt.".format(len(questions)))
    
    # Append the reply ID for test question:
    question, reply, comment, comment_id, tq_reply_id, subreddit = tq
    with open(path+'data/published_TQ_ids.txt', 'a') as f:
        f.write("%s\n" % tq_reply_id)
    logging.info("Added 1 ID to published_TQ_ids.txt: {}".format(tq_reply_id))    
    
    # Read XML:
    with open(path+'XMLs/task_layout.xml', 'r') as f:
        question_xml = f.read()
        
    # Add values for placeholder variables:
    for counter, row in enumerate(hit_questions):
        question, reply, comment, comment_id, reply_id, subreddit = row        
        # Remove non ASCII to avoid XML errors:
        question = ''.join([i if ord(i) < 128 else '' for i in question])
        reply = ''.join([i if ord(i) < 128 else '' for i in reply])
        comment = ''.join([i if ord(i) < 128 else '' for i in comment])
        
        question_xml = question_xml.replace('${reply_id_'+str(counter+1)+'}', reply_id)
        question_xml = question_xml.replace('${comment_text_'+str(counter+1)+'}', comment)
        question_xml = question_xml.replace('${reply_text_'+str(counter+1)+'}', reply)
        question_xml = question_xml.replace('${question_'+str(counter+1)+'}', question)
    
    return question_xml


def publish_hit(hit_type_id, n_hits, N=9):
    """
    Creates and publishes HITs on MTurk for the Workers to complete. If there is an error, saves the HIT's XML that caused it.
    
    Parameters
    ----------
    hit_type_id: str
        ID of the HIT Type created in create_HIT_type.py (saved in /pickles/)
    n_hits: int
        Number of HITs in the batch
    N: int (optional)
        Number of real questions per hit (excluding the secret Test Question). Default: 9 questions (+1 secret test question)
    """
    for n in range(n_hits):
        logging.info("WORKING ON HIT #%s", n+1)
        question_xml = get_HIT_xml(N)
        try:
            new_hit = mturk.create_hit_with_hit_type(
                            HITTypeId=hit_type_id,
                            MaxAssignments=5,
                            LifetimeInSeconds=2592000, # 30 days
                            Question=question_xml)
            logging.info("SUCCESS: HIT %s created.\n\n", n+1)
        except:
            fname = time.strftime("%Y_%m_%d_%H_%M_%S") + '.xml'
            with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/error_XMLs/'+fname, 'w') as f:
                f.write(question_xml)
            logging.critical("XML Error: HIT {} could not be published. See the XML in error_XMLs/{}".format(n+1, fname))

if __name__ == '__main__':    
    # Connect to MTurk Requester:
    access_keys = get_access_keys()
    mturk = boto3.client('mturk',
       aws_access_key_id = access_keys[0],
       aws_secret_access_key = access_keys[1],
       region_name='us-east-1',
       endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com') # DELETE this line for real AMT
    print "Available Balance in MTurk account: ", mturk.get_account_balance()['AvailableBalance']

    # Logging configuration:
    logging.basicConfig(filename='/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/logs/batch-3/MTurk_publish_HITs_batch3.log', filemode='a', format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

    # Create Qualification Type and HIT Type:
    qual_ID = create_qualification()
    hit_type_id = create_HIT_type(qual_ID)
    dic = {'QualificationTypeId': qual_ID, 'HITTypeId': hit_type_id} # Pickle the IDs just in case:
    pickle_fname = 'mturk_hit_ids_batch3.pickle'
    with open('/home/ndg/users/sbagga1/unpalatable-questions/pickles/'+pickle_fname, 'w') as f:
        pickle.dump(dic, f)
    print "Qualification Type and HIT Type have been created successfully.\nIDs in {} and log.".format(pickle_fname)
    
    GOLD_TQs = pd.read_csv('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/TestQuestions.csv', lineterminator='\n')['reply_id'].tolist()
    
    print("There are {} Test Questions in total.".format(len(GOLD_TQs)))
    
    # Publish HITs:
    publish_hit(hit_type_id, n_hits=700)