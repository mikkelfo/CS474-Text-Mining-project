# CS474-Text-Mining-project
Term project for CS474: Text Mining

[Google colab file for run through](https://colab.research.google.com/drive/1_8h8pZ5A0lb_MiyY5pOwc-F1UaY52IZZ)

## API Documentation
We've used 3 different libraries throughout the project, sci-kit learn, NLTK and spaCy.

### Sci-kit learn
Sci-kit learn was used throughout the entirety of the project using functionality such as
#### Vectorization
[CountVectorizer(max_features, max_df, ngram_range)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for counting terms
[TfidfVectorizer(max_features, min_df, max_df)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) for TFIDF matrix

#### Clustering
[DBSCAN(eps, min_samples)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
[MiniBatchKMeans(n_clusters)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)

#### LSA
[TrundcatedSVD(n_components, algorithm, n_iter, random_state)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

#### Similarity measures
[cosine similarity(document1, document2)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

