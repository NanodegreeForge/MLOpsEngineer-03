from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_metadata = os.path.join(os.getcwd(), config['output_folder_path'], "ingestedfiles.txt") 
output_model_path = config['output_model_path']
model_pkl_file = config['model_pkl_file']
model_score_file = config['model_score_file']
model_path = os.path.join(os.getcwd(), output_model_path, model_pkl_file) 
score_path = os.path.join(os.getcwd(), output_model_path, model_score_file)
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    if not os.path.exists(prod_deployment_path):  
        os.mkdir(prod_deployment_path)    
    shutil.copy(model_path, prod_deployment_path)
    shutil.copy(score_path, prod_deployment_path)
    shutil.copy(dataset_metadata, prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()

        

