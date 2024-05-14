from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
output_model_path = config['output_model_path']
model_pkl_file = config['model_pkl_file']
model_score_file = config['model_score_file']
model_path = os.path.join(output_model_path, model_pkl_file) 
score_path = os.path.join(output_model_path, model_score_file)



#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model = pickle.load(open(model_path, 'rb'))
    df_test = pd.read_csv(test_data_path)
    df_test = df_test.drop(["corporation"], axis=1)

    x1, x2, y1, y2 = train_test_split(df_test, df_test["exited"])
    model.fit(x1, y1)
    y_prediction = model.predict(x2)
    score = metrics.f1_score(y2, y_prediction)

    latest_score_path = os.path.join(
        os.getcwd(),
        output_model_path,
        'latestscore.txt'
    )
    latest_score = open(score_path, 'w')
    latest_score.write(str(score))

    return score

def main():
    score_model()

if __name__ == '__main__':
    main()
