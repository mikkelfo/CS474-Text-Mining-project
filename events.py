from collections import Counter

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import normalize

def extract_entities(df):
    import spacy
    nlp = spacy.load("en_core_web_lg")

    ents = df['edited'].apply(lambda x: list(nlp(x).ents))

    return ents

def clustering(matrix, df):
    np.random.seed(5)  # For the same results

    clusters = DBSCAN(eps=0.5, min_samples=4, metric="cosine").fit(matrix)
    labels = list(zip(*sorted(Counter(clusters.labels_).items(), key=lambda k: k[1], reverse=True)))[0]
    events = []
    for index in labels:
        if index != -1:
            indices = list(np.where(clusters.labels_ == index))[0]
            event = df[df.index.isin(indices)]
            events.append(event)

    return events

def group_by_n_date(df, n):
    df['time'] = pd.to_datetime(pd.to_datetime(df['time']).apply(lambda x: x.date()))
    freq = str(n)+'D'
    groups = []
    for _, grp in df.groupby(pd.Grouper(key="time", freq=freq)):
        if len(grp) != 0:
            groups.append(grp)

    return groups

# Finds title based on most similar body text
def find_title(event):
    titles = event['body'].apply(lambda x: normalize(x))

    titles_vector = TfidfVectorizer().fit_transform(titles)

    N = titles_vector.shape[0]
    sum_ = {}
    for i in range(N):
        sum_[i] = 0
        for j in range(N):
            if i != j:
                sum_[i] += cosine_similarity(titles_vector[i], titles_vector[j])
    for i in range(N):
        if N != 1:
            sum_[i] /= N - 1
    title_index = sorted(sum_.items(), key=lambda k: k[1], reverse=True)[0][0]

    event_title = event.reset_index()['title'].iloc[title_index]

    return event_title

def sort_by_date(events):
    timeline = []
    for event in events:
        timeline.append((event['time'].iloc[len(event)//2], event))

    timeline = list(zip(*sorted(timeline, key=lambda k: k[0])))[1]
    return timeline

def construct_timeline(events):
    timeline = sort_by_date(events)
    print("Event:")
    for event in timeline:
        print(" -> " + find_title(event))


def detailed_information(events, ents):
    timeline = sort_by_date(events)

    for event in timeline:
        print("Title:        " + find_title(event))
        print("Time:         " + find_time(event))
        print("Person:       " + ', '.join(find(event, ents, 'PERSON')))
        print("Organization: " + ', '.join(find(event, ents, 'ORG')))
        print("Place:        " + ', '.join(find(event, ents, 'GPE')))
        print()

def find_time(event):
    return event['time'].iloc[len(event)//2].strftime("%Y-%m-%d")

# valid labels include 'PERSON', 'ORG', 'GPE'
def find(df, ents, label):
    ents = ents[df.index]
    items = {}
    for doc in ents:
        for ent in doc:
            if ent.label_ == label:
                if ent.text in items:
                    items[ent.text] += 1
                else:
                    items[ent.text] = 1

    if remove_dupes(list(list(zip(*sorted(items.items(), key = lambda k: k[1], reverse=True))))):
        return remove_dupes(list(list(zip(*sorted(items.items(), key = lambda k: k[1], reverse=True)))[0])[:8])
    else:
        return []


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
