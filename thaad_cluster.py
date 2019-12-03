def thaad():
    from main import load_from_files
    dbscan_2015, dbscan_2016, dbscan_2017, kmeans_2015, kmeans_2016, kmeans_2017 = load_from_files()

    import pandas as pd
    thaad = pd.concat([dbscan_2017[4], dbscan_2016[1], dbscan_2015[0]], ignore_index = True)
    
    from preprocess import remove_stopwords, remove_punctuation
    thaad['edited'] = thaad['body']
    thaad['edited'] = thaad['edited'].apply(lambda x: x.lower())
    thaad['edited'] = thaad['edited'].apply(lambda x: x.strip())
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_stopwords(x))
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_punctuation(x))

    return thaad