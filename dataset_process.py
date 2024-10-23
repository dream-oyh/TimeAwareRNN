import os

import pandas as pd

files_list = os.listdir("dataset")
num_list = []
for file in files_list:
    dir = "dataset/" + file
    df = pd.read_csv(dir)
    num_list.append(df.shape[0])
print(num_list)