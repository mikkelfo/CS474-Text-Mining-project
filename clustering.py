from sklearn.cluster import DBSCAN, MiniBatchKMeans
import numpy as np
from datetime import datetime
from operator import itemgetter
import csv

def cluster(vectors, method=""):
    np.random.seed(5)  # For the same results
    # Asserts cluster method
    if method.lower() == "dbscan":
        print("Clustering with: DBSCAN(eps=0.85, min_samples=5)")
        clusters = DBSCAN(eps=0.85, min_samples=5).fit(vectors)

    elif method.lower() == "kmeans":
        print("Clustering with: KMeans(n_clusters=200)")
        clusters = MiniBatchKMeans(n_clusters=200).fit(vectors)
    else:
        raise NameError("Method name must be either 'dbscan' or 'kmeans'")

    return clusters

def ranking(df, labels):
    c1, c2, c3 = 2, 1, 0.5
    size, authors, time = [], [], []

    if -1 in labels:
        noise = 1
    else:
        noise = 0

    for i in range(0, len(set(labels))-noise):
        index = list(np.where(labels == i))[0]

        size.append(len([x for x in labels if x == i]))
        authors.append(df['author'][df['author'].index.isin(index)].nunique())

        if len(index) > 1:
            df_time = df['time'][df['time'].index.isin(index)]
            x1, x2 = datetime.strptime(df_time.iloc[0].split()[0], "%Y-%m-%d"), datetime.strptime(
                df_time.iloc[-1].split()[0], "%Y-%m-%d")
            duration = x1 - x2
            time.append(duration.days)
        else:
            time.append(1)

    result = {}
    norm_size = norm(size)
    norm_authors = norm(authors)
    norm_time = norm(time)
    for i in range(0, len(set(labels))-noise):
        result[i] = c1 * norm_size[i] + c2 * norm_authors[i] + c3 * norm_time[i]
    results = sorted(result.items(), key=itemgetter(1), reverse=True)
    return list(zip(*results))[0][0:10]

def indices(labels_, cluster_indices, filename):
    with open(filename, "a") as f:
        wr = csv.writer(f)
        list_of_lists = []
        for index in cluster_indices:
            list_of_lists.append(list(np.where(labels_ == index))[0])
        wr.writerows(list_of_lists)

def norm(l):
    l = np.array(l)
    return (l-np.min(l))/np.ptp(l)

