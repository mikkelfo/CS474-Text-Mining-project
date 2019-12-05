from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import normalize


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

def event_clustering(vectors, df):
    clusters = DBSCAN(eps=0.5, min_samples=4, metric="cosine").fit(vectors)
    labels = list(zip(*sorted(Counter(clusters.labels_).items(), key=lambda k: k[1], reverse=True)))[0]
    events = []
    for index in labels:
        if index != -1:
            indices = list(np.where(clusters.labels_ == index))[0]
            event = df[df.index.isin(indices)]
            events.append(event)

    return events


# Finds title based on most similar body text
def find_title(event):
    titles = event['body'].apply(lambda x: normalize(x))

    titles_vector = TfidfVectorizer(min_df=3, max_df=0.9).fit_transform(titles)

    N = titles_vector.shape[0]
    sum_ = {}
    for i in range(N):
        sum_[i] = 0
        for j in range(N):
            if i != j:
                sum_[i] += cosine_similarity(titles_vector[i], titles_vector[j])
    for i in range(N):
        sum_[i] /= N - 1
    title_index = sorted(sum_.items(), key=lambda k: k[1], reverse=True)[0][0]

    event_title = event.reset_index()['title'].iloc[title_index]

    return event_title


# Finds the median time
def find_time(event):
    time = event['time'].apply(lambda x: x.split()[0])
    return time.iloc[len(time) // 2]

# valid labels include 'PERSON', 'ORG', 'GPE'
def find(ents, label):
    items = {}
    for doc in ents:
        for ent in doc:
            if ent.label_ == label:
                if ent.text in items:
                    items[ent.text] += 1
                else:
                    items[ent.text] = 1

    return remove_dupes(list(list(zip(*sorted(items.items(), key = lambda k: k[1], reverse=True)))[0])[:8])


# Helper functions
def remove_dupes(l):
    l.sort(key=lambda x: len(x), reverse=True)
    output = []
    for x in l:
        if not is_in(x, output):
            output.append(x)
    return output


def is_in(word, check):
    for x in check:
        if word in x:
            return True
    return False
