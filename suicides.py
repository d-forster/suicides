'''
This file contains all of the code dealing with data analysis + representation.

TODO 
'''

# Package imports
import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt

# Reading dataset
train_file_path = "./input/master.csv"
dataset_df = pd.read_csv(train_file_path)
print("Loading dataset CSV...")
print("Full dataset shape is " + dataset_df.shape)

