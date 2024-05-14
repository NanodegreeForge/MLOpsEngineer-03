import io
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import dataframe_summary, execution_time, percent_na
import json
import os

from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

test_data = os.path.join(os.getcwd(), config['test_data_path'], "testdata.csv") 
prod_deployment_path = config['prod_deployment_path']
model_pkl_file = config['model_pkl_file']
model_path = os.path.join(os.getcwd(), prod_deployment_path, model_pkl_file) 
prediction_model = pickle.load(open(model_path, 'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    csv_file = request.args.get('inputdata')
    df = pd.read_csv(os.path.join(os.getcwd(), csv_file)).drop(['corporation'], axis=1)
    #call the prediction function you created in Step 3
    prediction = prediction_model.predict(df)
    return jsonify(prediction.tolist()), 200

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    df = pd.read_csv(test_data)
    #check the score of the deployed model
    return jsonify(score_model()), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return jsonify(dataframe_summary()), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    return jsonify({
        "execution_time": execution_time(),
        "percent_na": percent_na()
    })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
