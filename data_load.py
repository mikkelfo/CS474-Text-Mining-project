import pandas as pd
import os


def load(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        with open(path + file, "r") as f:
            data = pd.read_json(f)
            df = df.append(data, ignore_index=True)
    df = fix_headers(df)
    return df

# There was whitespace in the headers
def fix_headers(df):
    new_headers = []
    for header in df.columns:
        new_headers.append(header.strip())
    df.columns = new_headers
    
    return df
