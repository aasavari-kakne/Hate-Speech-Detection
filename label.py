import pandas as pd
import string
import os
import numpy as np
import json
import random
from tqdm import tqdm

# global variables
DATA_DIR = '/home/ubuntu/CS224u_Final_Project/data'
output_path = os.path.join(DATA_DIR, 'hand_label.csv')
text_path = os.path.join(DATA_DIR, 'soft_labels_14_mil.txt')

# step 1 : choose random indices
indices = random.sample(list(np.arange(1400000)), 120)

# step 2 : read txt file as dataframe
pred_data = pd.read_csv(text_path, delimiter="\t", header=None, names=['processed_tweet', 'prediction', 'str_id'])

# step 3 : read json file as a dataframe
sample = pred_data.iloc[indices, :]

# step 4 : create a dataframe and write to csv file
sample.to_csv(output_path, index=False)


