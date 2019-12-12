"""
Author: Sunyam

This file retrieves the results for HITs that Workers have completed and saves them in a TSV along with the Worker's final accuracy.
"""
import boto3
import os
import xmltodict
import csv
import pickle
import sys


def get_worker_accuracy_assignments(worker_id, monitor_logger):
    """
    Reads the log file and returns the given worker's final accuracy along with the total number of assignments they
    completed (based on which that accuracy was calculated). Note that it is possible that their qualification was 
    revoked later, and they submitted another assignment which would have been rejected. This assignment count does
    not include those rejected assignments.
    
    Parameters
    ----------
    worker_id: str
    monitor_logger: path to Logger file (for batch 1 or 2 or 3)

    Returns
    -------
    accuracy: float
    total_assignments: int
    """
    with open(monitor_logger, 'r') as f:
        data = f.readlines()
        
    last_row = ''
    for row in data:
        elements = row.split('-')
        if elements[0].strip() == 'INFO' and elements[-1].strip().startswith('Worker '+worker_id):
            last_row = row
            
    accuracy = float(last_row.split('=')[-1].strip())
    total_assignments = last_row.split('|')[0].split()[-2].strip()
    return accuracy, int(total_assignments)


def write_to_tsv(assignments_for_HIT, monitor_logger):
    """
    Writes the annotated data into a TSV.
    Columns = "Reply ID" "Label" "Worker ID" "Accuracy"
    """   
    num_results = assignments_for_HIT['NumResults']
    print "There are {} assignments for this HIT:".format(num_results)
    
    for assignment in assignments_for_HIT['Assignments']:
        assignment_id = assignment['AssignmentId']
        if assignment_id in ALL_ASSGN_IDS:
            print "Assignment already extracted..skipping"
            continue
            
        ALL_ASSGN_IDS.add(assignment_id)
        worker_id = assignment['WorkerId']
        accuracy, number_of_assignments = get_worker_accuracy_assignments(worker_id, monitor_logger)
        print "Extracting accuracy from", monitor_logger.split('/')[-2]
        
        xml_doc = xmltodict.parse(assignment['Answer'])
        for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
            q_id = answer_field['QuestionIdentifier']
            ALL_Q_IDS.append(q_id)
            
            ans = answer_field['SelectionIdentifier']
            with open(TSV_PATH, 'a') as f:
                f.write(q_id + '\t' + ans + '\t' + worker_id + '\t' + str(accuracy) + '\t' + str(number_of_assignments) + '\n')


def get_access_keys():
    with open('/home/credentials.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header row
        for row in csv_reader:
            access_keys = row
    return access_keys


def cache_annotated_data(hits_response):
    """
    Caches the annotated data to a TSV. (Note: only saves the 'Approved' assignments)
    """
    assgn_paginator = mturk.get_paginator('list_assignments_for_hit')
    
    for hit in hits_response['HITs']:
        if hit['HITTypeId'] == '38NEE7DWQJ0X2NR33ERK8L5QQJNT0D': # Batch 1
            monitor_logger = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/logs/batch-1/MTurk_monitor_HITs_batch1.log'
        elif hit['HITTypeId'] == '3O8UG586S9LVHW0QO995TOURDO3497': # Batch 2
            monitor_logger = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/logs/batch-2/MTurk_monitor_HITs_batch2.log'
        elif hit['HITTypeId'] == '37HWZYGM8VXBF3EADGXC9DYY8P39YY': # Batch 3
            monitor_logger = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/logs/batch-3/MTurk_monitor_HITs_batch3.log'            
        else:
            sys.exit("Bug in code")
            
        assign_iterator = assgn_paginator.paginate(HITId=hit['HITId'], AssignmentStatuses=['Approved'])
        for response in assign_iterator:
            if response['NumResults'] > 0:
                write_to_tsv(response, monitor_logger)
            else:
                pass
#                 print "No assignments approved yet | Not calling write_to_tsv()"


def main():
#     hit_paginator = mturk.get_paginator('list_reviewable_hits')
#     hit_iterator = hit_paginator.paginate(HITTypeId=HIT_TYPE_ID, Status='Reviewable')

#     hit_paginator = mturk.get_paginator('list_hits_for_qualification_type')
#     hit_iterator = hit_paginator.paginate(QualificationTypeId=QUALIFICATION_TYPE_ID)

    hit_paginator = mturk.get_paginator('list_hits')
    hit_iterator = hit_paginator.paginate()
    
    for response in hit_iterator:
        if response['NumResults'] > 0:
            print "\nThere are {} HITs to process (in this page)".format(response['NumResults'])
            cache_annotated_data(response)

        else:
            pass
#             print "No reviewable HITs available: ", response


if __name__ == '__main__':
    ALL_Q_IDS = [] # make sure we have all Reply IDs
    ALL_ASSGN_IDS = set() # make sure we don't double count assignments
    
    TSV_PATH = '/home/ndg/users/sbagga1/unpalatable-questions/crowdsourcing/annotations/annotations_batch_1_2_3.tsv'    
    if os.path.exists(TSV_PATH):
        sys.exit("Error: TSV already exists: " + str(TSV_PATH))
        
    with open(TSV_PATH, 'a') as f:
        f.write('Reply ID\tLabel\tWorker ID\tAccuracy\t# Assignments\n')

    # Connect to MTurk Requester:
    access_keys = get_access_keys()
    mturk = boto3.client('mturk',
       aws_access_key_id = access_keys[0],
       aws_secret_access_key = access_keys[1],
       region_name='us-east-1',)
#        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com') # DELETE this line for real AMT
    print "Available Balance in MTurk account: ", mturk.get_account_balance()['AvailableBalance']
        
#     HIT_TYPE_ID = ''
#     QUALIFICATION_TYPE_ID = ''
    main()
    print "\n\nDONE!\nTotal Assignments = {}\nTotal Reply IDs = {} | Unique Reply IDs = {}".format(len(ALL_ASSGN_IDS), len(ALL_Q_IDS), len(set(ALL_Q_IDS)))