from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np

# [(0.45, 3), (0.5, 4), (0.45, 5)]
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
