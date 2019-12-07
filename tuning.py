from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np

def try_events(params, vectors, thaad):
    candidates = []
    for (eps, min_samples) in params:
        clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)

        labels = list(zip(*sorted(Counter(clusters.labels_).items(), key=lambda k: k[1], reverse=True)))[0]
        events = []
        for index in labels:
            if index != -1:
                indices = list(np.where(clusters.labels_ == index))[0]
                event = thaad[thaad.index.isin(indices)]
                events.append(event)

        candidates.append(events)

    return candidates

def event_clustering(vectors):
    # Event clustering parameter tuning

    from sklearn.cluster import DBSCAN
    import pandas as pd
    import numpy as np

    np.random.seed(5)

    rang = range(40, 60, 5)
    minrang = range(3, 8)

    headers = [str(x / 100) for x in rang]
    data = []

    for eps in rang:
        eps /= 100
        t = []
        for min_samples in minrang:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
            t.append(len([x for x in clusters.labels_ if x != -1]))
        data.append(t)
    data = np.array(data)
    df = pd.DataFrame(data.transpose(), columns=headers, index=minrang)
    print("Clustering test for election cluster")
    print("Columns = eps, row = min_samples")
    print("values are number of points")
    print(df.to_string())
    print()

    data = []
    for eps in rang:
        eps /= 100
        t = []
        for min_samples in minrang:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
            t.append(len(set(clusters.labels_)) - 1)
        data.append(t)
    data = np.array(data)
    df = pd.DataFrame(data.transpose(), columns=headers, index=minrang)
    print("Clustering test for election cluster")
    print("Columns = eps, row = min_samples")
    print("values are # of clusters")
    print(df.to_string())
    print()

    data = []
    for eps in rang:
        eps /= 100
        t = []
        for min_samples in minrang:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
            t.append(str(round(len([x for x in clusters.labels_ if x != -1]) / len(clusters.labels_), 2)))
        data.append(t)
    data = np.array(data)
    df = pd.DataFrame(data.transpose(), columns=headers, index=minrang)
    print("Clustering test for election cluster")
    print("Columns = eps, row = min_samples")
    print("values are non-noise/noise ratio")
    print(df.to_string())
    print()

    data = []
    for eps in rang:
        eps /= 100
        t = []
        for min_samples in minrang:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
            x = []
            for i in range(0, len(set(clusters.labels_))):
                x.append(len([x for x in clusters.labels_ if x == i]))
            t.append(sorted(x, reverse=True)[:5][-1])
        data.append(t)
    data = np.array(data)
    df = pd.DataFrame(data.transpose(), columns=headers, index=minrang)
    print("Clustering test for election cluster")
    print("Columns = eps, row = min_samples")
    print("values are 5th lowest cluster size")
    print(df.to_string())
    print()

    data = []
    for eps in rang:
        eps /= 100
        t = []
        for min_samples in minrang:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
            x = []
            for i in range(0, len(set(clusters.labels_))):
                x.append(len([x for x in clusters.labels_ if x == i]))
            t.append(sorted(x, reverse=True)[0])
        data.append(t)
    data = np.array(data)
    df = pd.DataFrame(data.transpose(), columns=headers, index=minrang)
    print("Clustering test for election cluster")
    print("Columns = eps, row = min_samples")
    print("values are biggest cluster size")
    print(df.to_string())
