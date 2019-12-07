# CS474-Text-Mining-project
Term project for CS474: Text Mining

[Google colab file for run through](https://colab.research.google.com/drive/1_8h8pZ5A0lb_MiyY5pOwc-F1UaY52IZZ)

## API Documentation for libaries used
We've used 3 different libraries throughout the project, sci-kit learn, NLTK and spaCy.

### [Scikit learn](http://scikit-learn.github.io/stable)
Sci-kit learn was used throughout the entirety of the project using functionality such as
#### Vectorization
[CountVectorizer(max_features, max_df, ngram_range)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)<br/>
[TfidfVectorizer(max_features, min_df, max_df)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

#### Clustering
[DBSCAN(eps, min_samples)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)<br/>
[MiniBatchKMeans(n_clusters)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)

#### LSA
[TrundcatedSVD(n_components, algorithm, n_iter, random_state)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

#### Similarity measures
[cosine similarity(document1, document2)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

### [spaCy](https://spacy.io/)
spaCy was primarily used for Named Entity Recognition, using the model 'en_core_web_lg', an english CNN with tons of features.

Entities was obtained simply by loading 'en_core_web_lg' into a variable *nlp* then calling *nlp(text).ents*

### [Natural Language Toolkit (NLTK)](https://www.nltk.org/)
NLTK was only used in preprocessing, namely to remove punctionations, remove stop words, tokenize and lemmatize our articles. Lastly, it was used as a POS-tagger to extract nouns.

