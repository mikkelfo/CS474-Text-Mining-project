import pandas as pd
import os

temp = pd.DataFrame()

path = 'articles/'
for file in os.listdir(path):
    with open(path + file, "r") as f:
        data = pd.read_json(f)
        temp = temp.append(data, ignore_index = True)

    