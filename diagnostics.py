
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import pandas
import subprocess
from sklearn.model_selection import train_test_split
from ingestion import merge_multiple_dataframe
from training import train_model
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

training_data = os.path.join(os.getcwd(), config['output_folder_path'], config['final_data_file']) 
test_data = os.path.join(os.getcwd(), config['test_data_path'], "testdata.csv") 
prod_deployment_path = config['prod_deployment_path']
model_pkl_file = config['model_pkl_file']
model_path = os.path.join(os.getcwd(), prod_deployment_path, model_pkl_file) 


##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(model_path, "rb"))
    df_test = pandas.read_csv(test_data).drop(['corporation'], axis=1)
    x1, x2, y1, y2 = train_test_split(df_test, df_test["exited"])
    y_prediction = model.predict(x2)
    return y_prediction

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df_final = pandas.read_csv(training_data).drop(['corporation'], axis=1)
    summary = df_final.describe()
    return summary.values.tolist()

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timing = []
    # training.py
    timer = timeit.default_timer()
    merge_multiple_dataframe()
    timing.append(timeit.default_timer() - timer)

    # ingestion.py
    timer = timeit.default_timer()
    train_model()
    timing.append(timeit.default_timer() - timer)

    return timing

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdated = subprocess.check_output(['python', '-m', 'pip', 'list', '--outdated'])
    return outdated

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
