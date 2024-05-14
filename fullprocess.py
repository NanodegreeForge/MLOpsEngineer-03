

import sys
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
final_data_file = config['final_data_file']
ingested_files = os.path.join(os.getcwd(), config['output_folder_path'], "ingestedfiles.txt") 
output_model_path = config['output_model_path']
model_pkl_file = config['model_pkl_file']
model_score_file = config['model_score_file']
model_path = os.path.join(os.getcwd(), output_model_path, model_pkl_file) 
score_path = os.path.join(os.getcwd(), output_model_path, model_score_file)

if __name__ == '__main__':
    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_list = [file.strip() for file in open(ingested_files, 'r').readlines()]
    files_list = os.listdir(os.path.join(os.getcwd(), input_folder_path))

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    arent_ingested = [file for file in files_list if file not in ingested_list]

    ##################Deciding whether to proceed, part 1
    #if not found new data do end the process here
    if (len(arent_ingested) == 0):
        sys.exit()

    #if you found new data, you should proceed.
    df = ingestion.merge_multiple_dataframe()
    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    prev_score = float(open(score_path, 'r').read())
    new_score = scoring.score_model()

    ##################Deciding whether to proceed, part 2
    #if not found model drift do end the process here
    if new_score == prev_score:
        sys.exit()

    #if you found model drift, you should proceed.
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    ingestion.main()
    training.main()
    scoring.main()
    deployment.main()


    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    diagnostics.main()
    reporting.main()






