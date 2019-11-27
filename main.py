from data_load import load
from preprocess import preprocess
from clustering import cluster, ranking, indices
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

def write_to_files():

    df = load('articles/')
    df = preprocess(df)

    tfidf = TfidfVectorizer(max_features=1000, min_df=10, max_df=0.9)

    df['year'] = df['time'].str[:4]
    year2015, year2016, year2017 = df[df['year'] == '2015'], df[df['year'] == '2016'], df[df['year'] == '2017']

    # Resets the files
    with open("dbscan.csv", "w") as f:
        pass
    with open("kmeans.csv", "w") as f:
        pass

    for df_year in [year2015, year2016, year2017]:
        vectors = tfidf.fit_transform(df_year['edited'])

        clusters = cluster(vectors, method="dbscan")
        top10 = ranking(df, clusters.labels_)
        indices(clusters.labels_, top10, "dbscan.csv")

        clusters = cluster(vectors, method="kmeans")
        top10 = ranking(df, clusters.labels_)

        indices(clusters.labels_, top10, "kmeans.csv")

def load_from_files():
    df = load('articles/')

    with open("dbscan.csv") as f:
        fr = csv.reader(f)
        reader = list(fr)
        cluster_indices = [[int(x) for x in row] for row in reader]
    db = []
    for row in cluster_indices:
        db.append(df[df.index.isin(row)])
    dbscan_2015 = db[0:10]
    dbscan_2016 = db[10:20]
    dbscan_2017 = db[20:30]

    with open("kmeans.csv") as f:
        fr = csv.reader(f)
        reader = list(fr)
        cluster_indices = [[int(x) for x in row] for row in reader]
    km = []
    for row in cluster_indices:
        km.append(df[df.index.isin(row)])
    kmeans_2015 = km[0:10]
    kmeans_2016 = km[10:20]
    kmeans_2017 = km[20:30]

    return dbscan_2015, dbscan_2016, dbscan_2017, kmeans_2015, kmeans_2016, kmeans_2017
