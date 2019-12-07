from main import load_dbscan
import pandas as pd
from preprocess import remove_stopwords, remove_punctuation, extract_nouns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


def title_extraction(df):
    for index in range(len(df)):

        temp = df[index]['body'].apply(extract_nouns)

        cv = CountVectorizer(max_df=0.5, max_features=1000, ngram_range=(1, 3))
        X = cv.fit_transform(temp)

        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(n_components=1, algorithm='randomized', n_iter=100, random_state=122)
        svd_model.fit(X)

        terms = cv.get_feature_names()

        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
            print("Cluster " + str(index) + ": " + str(list(t[0] for t in sorted_terms)))

def thaad():
    dbscan_2015, dbscan_2016, dbscan_2017 = load_dbscan()

    thaad = pd.concat([dbscan_2017[4], dbscan_2016[1], dbscan_2015[0]], ignore_index=True)

    thaad['edited'] = thaad['body']
    thaad['edited'] = thaad['edited'].apply(lambda x: x.strip())
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_stopwords(x))
    thaad['edited'] = thaad['edited'].apply(lambda x: remove_punctuation(x))

    return thaad

def election():
    _, _, dbscan_2017 = load_dbscan()

    election = dbscan_2017[0].reset_index(drop=True)

    election['edited'] = election['body']
    election['edited'] = election['edited'].apply(lambda x: x.strip())
    election['edited'] = election['edited'].apply(lambda x: remove_stopwords(x))
    election['edited'] = election['edited'].apply(lambda x: remove_punctuation(x))

    return election
