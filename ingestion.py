import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
final_data_file = config['final_data_file']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    files_list = os.listdir(os.path.join(os.getcwd(), input_folder_path))

    files_path = glob.glob(
        os.path.join(
            os.getcwd(),
            input_folder_path,
            '*.csv'))
    # Concat DataFrame
    df = pd.concat([pd.read_csv(f) for f in files_path], ignore_index=True)
    # Dedup
    process_df = df.drop_duplicates()
    # Saving DF to output
    if not os.path.exists(output_folder_path):  
        os.mkdir(output_folder_path)    
    process_df.to_csv(os.path.join(os.getcwd(), output_folder_path, final_data_file), index=False)

    with open(os.path.join(os.getcwd(), output_folder_path, "ingestedfiles.txt"), 'w') as f:
        for file in files_list:
            f.write('{}\n'.format(file))

    return process_df




if __name__ == '__main__':
    merge_multiple_dataframe()
