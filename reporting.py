import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os

from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data = os.path.join(os.getcwd(), config['test_data_path'], "testdata.csv") 
cm_path = os.path.join(config['output_model_path'], "confusionmatrix.png")

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(test_data).drop(['corporation'], axis=1)
    y = df['exited'].values
    y_prediction = model_predictions()

    cm = metrics.confusion_matrix(y,y_prediction)
    metrics.ConfusionMatrixDisplay(cm, display_labels=['0', '1']).plot()
    plt.savefig(cm_path)


def main():
    score_model()

if __name__ == '__main__':
    score_model()
