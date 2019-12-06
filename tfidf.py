def tokenizer(x):
    text = []
    for ent in x:
        text.append((ent.text.lower(), ent.label_.lower()))
    return text


# Not used
def tfidf(ents):
    import math
    import pandas as pd
    import scipy

    min_df = 3
    max_df = 0.9

    df = pd.DataFrame()
    for i in range(len(ents)):
        for ent in ents.iloc[i]:
            df[(ent.text, ent.label_)] = [0] * len(ents)

    for ent in df.columns:
        t = []
        for i in range(len(ents)):
            if len(ents.iloc[i]) == 0:
                t.append(0)
            else:
                t.append(
                    len([x for x in ents.iloc[i] if (x.text, x.label_) == ent]) / (len(ents.iloc[i])))  # Term frequency
        df[ent] = t

    N = len(df)
    for ent in df.columns:
        doc_freq = df[ent].astype(bool).sum(axis=0)
        if doc_freq >= min_df and doc_freq / N <= max_df:  # min_df = 3, max_df = 0.9
            df[ent] *= math.log(N / doc_freq, 2)  # log2(N/df(t))
        else:
            df[ent] = 0

    vectors = scipy.sparse.csr_matrix(df.values)

    return df
