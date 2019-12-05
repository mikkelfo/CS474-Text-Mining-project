from main import load_dbscan
import pandas as pd
from preprocess import remove_stopwords, remove_punctuation

def thaad():
    dbscan_2015, dbscan_2016, dbscan_2017 = load_dbscan()

    thaad = pd.concat([dbscan_2017[4], dbscan_2016[1], dbscan_2015[0]], ignore_index=True)

    thaad['edited'] = thaad['body']
    thaad['edited'] = thaad['edited'].apply(lambda x: x.lower())
    thaad['edited'] = thaad['edited'].apply(lambda x: x.strip())
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_stopwords(x))
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_punctuation(x))

    return thaad

def election():
    _, _, dbscan_2017 = load_dbscan()

    election = dbscan_2017[0]

    election['edited'] = election['body']
    election['edited'] = election['edited'].apply(lambda x: x.lower())
    election['edited'] = election['edited'].apply(lambda x: x.strip())
    election['edited'] = election['edited'].apply(lambda x: remove_stopwords(x))
    election['edited'] = election['edited'].apply(lambda x: remove_punctuation(x))

    return election
