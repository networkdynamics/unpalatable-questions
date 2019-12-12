"""
Author: Sunyam

This file is for monitoring submissions and ensuring quality control.
- Monitors the submitted HITs (assignments).
- Updates Worker's score based on their answers to secret test questions.
- Revokes qualification if score falls below the threshold.
"""

import boto3
import time
import pickle
import csv
import sys
import logging
import pandas as pd
import xmltodict
from datetime import datetime
from collections import defaultdict


def get_gold_question_dict():
    """
    Returns a dictionary for the Test Questions (reads /data/TestQuestions.csv)
        key = reply_id; value = label ('yes_unpalatable'|'not_unpalatable')
    """
    df = pd.read_csv('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/TestQuestions.csv', lineterminator='\n')
    map_replyID_label = dict([tuple(x) for x in df[['reply_id', 'label']].values])
    return map_replyID_label


def get_published_test_questions():
    """
    Returns
    -------
    set
        reply IDs corresponding to already published Test Questions (including Qualification test).
    """
    with open('/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/data/published_TQ_ids.txt', 'r') as f:
        TQs_USED = f.read().splitlines()

    if len(TQs_USED) != len(set(TQs_USED)):
        logging.warning("Duplicate test questions in published_TQ_ids.txt;\
        Total count = {}; Unique count = {}".format(len(TQs_USED), len(set(TQs_USED))))

    TQs_USED = set(TQs_USED)
    logging.info("Number of unique TQs published: %s", len(TQs_USED))
    return TQs_USED


def get_qual_score(worker_id):
    """
    Returns the qualification score for the given Worker ID.
    Returns None if the call to the API does not go through (eg: Worker's qualification has been revoked)
    """
    try:
        r = mturk.get_qualification_score(QualificationTypeId=QUALIFICATION_TYPE_ID, WorkerId=worker_id)
        return r['Qualification']['IntegerValue']
    except:
        return None
    

def update_qualification_status(worker_id):
    """
    Revoke qualification status of the given Worker ID if their accuracy falls below the threshold. Also, notify them.
    Note: Accuracy is calculated dynamically using their current score and number of HITs they have submitted.
    """
    # Calculate percentage of answers they get correct:
    score = get_qual_score(worker_id)    
    if score == None: # Worker's qualification has already been revoked
        print "\nBUG BUG:: SHOULD NEVER GET HEREE....update_qual_status() if block"
        logging.critical('BUG IN CODE: Can not get here in update_qual_status()')
        return None
    
    total_assignments = len(map_workerID_assignments[worker_id])
    accuracy = 100 * (float(score) / (total_assignments + N_QUIZ)) # N_QUIZ: number of questions in the qualification test
    logging.info("Worker {} has completed {} assignments | Accuracy = {}".format(worker_id, total_assignments, accuracy))
    if accuracy > 100:
        logging.critical("BUG IN CODE: Accuracy is greater than 100 - %s", accuracy)
    
    if accuracy < THRESHOLD: # Disqualify them and notify:
        message = 'Hi! Since your real-time accuracy on the test questions has fallen below our predefined threshold, you will not be allowed to submit any more HITs. Thank you for contributing to our task!'
        mturk.disassociate_qualification_from_worker(WorkerId=worker_id, QualificationTypeId=QUALIFICATION_TYPE_ID, Reason=message)
#         mturk.create_worker_block(WorkerId=worker_id, Reason=message)
        print "In update_qual_status(): Revoked Worker {} because accuracy = {}".format(worker_id, accuracy)
        logging.warning("REVOKED Qualification for Worker {}".format(worker_id))
        
    else:
        print "In update_qual_status(): Worker {} is still doing well with accuracy = {}".format(worker_id, accuracy)
        pass

    
def handle_disqualified_submission(worker_id, hit_id, assignment_id, n_assignments):
    """
    This method is called if a disqualified Worker makes another submission. Can only happen if they had accepted a HIT before the disqualification processing completed (API latency lag).
    - Rejects assignment.
    - Logs the details.
    - Tries to create an additional assignment (NOTE: Not allowed to make more than 10).
    
    NOTE: After batch1 run, this was changed to approving assignments and waiting for the disqualification to be completed. Rejecting hurts AMT Workers badly.
    """
    mturk.approve_assignment(AssignmentId=assignment_id)
    logging.critical("Possibly Revoked Worker {} still contributing to the task | Approved his latest assignment {} | Assignment Count = {}".format(worker_id, assignment_id, len(map_workerID_assignments[worker_id])))

    # Commented after Batch 1 run:
#     message = 'Hi! Since your accuracy on the test questions has fallen below the predefined threshold, you will not be allowed to submit any more HITs. Unfortunately, we have to reject this assignment. Thank you for contributing to our task!'

#     mturk.reject_assignment(AssignmentId=assignment_id, RequesterFeedback=message)   
#     try: # try creating an additional assignment (Amazon gives an error if >10)
#         mturk.create_additional_assignments_for_hit(HITId=hit_id, NumberOfAdditionalAssignments=1)    
#         logging.critical("Possibly Revoked Worker {} still contributing to the task | Re-ordered 1 Additional Assignment | Rejected his latest assignment {} | Assignment Count = {}".format(worker_id, assignment_id, len(map_workerID_assignments[worker_id])))
        
#     except: # wasn't able to create an additional assignment
#         logging.critical("Possibly Revoked Worker {} still contributing to the task | Rejected his latest assignment {} | Assignment Count = {} | Additional Assignment NOT created".format(worker_id, assignment_id, len(map_workerID_assignments[worker_id])))

        
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

    
def process_assignments_for_HIT(assignments_for_HIT):
    """
    Processes all 'Submitted' assignments for a single HIT | 1 HIT: multiple assignments and multiple worker IDs.
    - Updates map_workerID_assignments dictionary (avoids double counting by keeping track of the set of assignment IDs)
    - Updates the worker's qualification score and qualification status.
    - Approves all assignments (but revoked Workers would not be allowed to submit any further HITs).
    """
    num_results = assignments_for_HIT['NumResults']
    print "In process_assignments_for_HIT()..there are {} results at time = {}".format(num_results, datetime.now())
    
    for assignment in assignments_for_HIT['Assignments']:
        hit_id = assignment['HITId']
        assignment_id = assignment['AssignmentId']
        worker_id = assignment['WorkerId']
        TQ_seen = 0 # Keep track of the Test Question per assignment

        # Update submission count for the Worker:
        if assignment_id in map_workerID_assignments[worker_id]:
            logging.error('Assignment {} already existed in map_workerID_assignments for Worker {}'.format(assignment_id, worker_id))
        map_workerID_assignments[worker_id].add(assignment_id)

        xml_doc = xmltodict.parse(assignment['Answer'])
        for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
            q_id = answer_field['QuestionIdentifier']
            ans = answer_field['SelectionIdentifier']

            current_score = get_qual_score(worker_id)
            if current_score == None: # Worker's qualification has already been revoked
                handle_disqualified_submission(worker_id, hit_id, assignment_id, len(map_workerID_assignments[worker_id]))
                return None

            if q_id in TQs_USED: # If this is a published test question
                TQ_seen += 1
                # Update qualification score
                if ans == map_replyID_label[q_id]: # Worker got the TQ right (add +1 to their score)
                    mturk.associate_qualification_with_worker(QualificationTypeId=QUALIFICATION_TYPE_ID, 
                                                              WorkerId=worker_id,
                                                              IntegerValue=current_score+1,
                                                              SendNotification=False)
                else: # Worker got the TQ wrong | Score is untouched
                    logging.info('Test Question Missed: {} | Worker: {}'.format(q_id, worker_id))

                # Now, update qualification status:
                update_qualification_status(worker_id)

            else: # Just a regular question
                pass
    
        # Approve assignment:
        mturk.approve_assignment(AssignmentId=assignment_id)
        logging.info("Approved Assignment = {} by Worker = {}".format(assignment_id, worker_id))
        if TQ_seen != 1: # Sanity check: making sure that Test Question appears exactly once
            logging.critical("BUG IN CODE: Assignment {} saw {} Test Questions!".format(assignment_id, TQ_seen))

            
def process_HITs(hits_response):
    """
    Processes all 'Submitted' assignments for all HITs.
    """
    for hit in hits_response['HITs']:
        try:
            assign_iterator = ASSGN_PAGINATOR.paginate(HITId=hit['HITId'], AssignmentStatuses=['Submitted'])
            for response in assign_iterator:
                if response['NumResults'] > 0:
                    print "# HIT ID =", hit['HITId'], "| Number of Assignments available:", hit['NumberOfAssignmentsAvailable']
                    process_assignments_for_HIT(response)
                else:
                    pass
            
        except Exception as e:
            print e
            time.sleep(2)
            save_assignment_counts()
           
                
def monitor_MTurk():
    """
    Loop through all HITs and monitor quality.
    """
    try:
        hit_iterator = HIT_PAGINATOR.paginate(QualificationTypeId=QUALIFICATION_TYPE_ID)
        for response in hit_iterator:
            if response['NumResults'] > 0:
                print "In this page, there are {} HITs to process.".format(response['NumResults'])
                process_HITs(response)
            else:
                print "\nNot calling process_HITs because: NumResults =", response['NumResults']

    except Exception as e:
        print e
        time.sleep(2)
        save_assignment_counts()
        
                  
def save_assignment_counts():
    """
    Pickles the map_workerID_assignments dictionary; need it to compute accuracy in the next batch.
    """
    fname = 'batch_3_worker_n_assignments.pickle'
    
    with open('/home/ndg/users/sbagga1/unpalatable-questions/pickles/'+fname, 'w') as out_file:
        pickle.dump(map_workerID_assignments, out_file)
    print("\nPICKLE DUMPED: {}.".format(fname))
    
    
if __name__ == '__main__':    
    # Connect to MTurk Requester:
    access_keys = get_access_keys()
    mturk = boto3.client('mturk',
       aws_access_key_id = access_keys[0],
       aws_secret_access_key = access_keys[1],
       region_name='us-east-1',)
#        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com') # DELETE this line for real AMT
    print "Available Balance in MTurk account: ", mturk.get_account_balance()['AvailableBalance']
    
    logging.basicConfig(filename='/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/logs/batch-3/MTurk_monitor_HITs_batch3.log', filemode='a', format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)
    
    with open('/home/ndg/users/sbagga1/unpalatable-questions/pickles/mturk_hit_ids_batch3.pickle', 'r') as f:
        type_ids = pickle.load(f)
        
    TQs_USED = get_published_test_questions() # Set of published test questions
    map_replyID_label = get_gold_question_dict() # Maps Test Question's reply ID to the correct label
    
    # Sanity check: make sure we have gold answers for all Test Questions
    if not TQs_USED.issubset(set(map_replyID_label.keys())):
        sys.exit("EXIT: Error in Test Question CSV!")
    
    QUALIFICATION_TYPE_ID = type_ids['QualificationTypeId']
    N_QUIZ = 10 # Number of questions in the qualification test
    THRESHOLD = 90 # Workers must maintain this % accuracy to stay qualified
    
    
    RE_RUN = False # Set to True ONLY if this is a re-run (or to track accuracy across batches using the same qual type)
    if RE_RUN:
        p_fname = 'batch_3_worker_n_assignments.pickle'
        print "It is a RE-RUN: initialising dictionary using {}".format(p_fname)
        with open('/home/ndg/users/sbagga1/unpalatable-questions/pickles/'+p_fname, 'r') as f2:
            map_workerID_assignments = pickle.load(f2)
    else:
        map_workerID_assignments = defaultdict(set) # Maps Worker ID to their set of submitted assignments
        logging.info('\n\nBEGIN MONITORING....')

    ASSGN_PAGINATOR = mturk.get_paginator('list_assignments_for_hit')
    HIT_PAGINATOR = mturk.get_paginator('list_hits_for_qualification_type')
    
    counter_i = 0
    while True:
        try:
            monitor_MTurk()
        except Exception as e:
            time.sleep(60)
            
        counter_i += 1
        if counter_i % 50 == 0:
            print "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPickling every 50 loops..\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            save_assignment_counts()