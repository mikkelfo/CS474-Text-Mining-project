# Team 5's term project for CS474: Text Mining

## API Documentation for code
We had problems running clustering algorithms on our computers (memory issues) and therefore switched to google colab. This means that we do not have an executable or main file (our main file consists of write/loading files), but instead a detailed google colab.

We've constructed a google colab file that runs through the entirety of our code, explaining the steps along the way. <br/>
[Google colab file for run through](https://colab.research.google.com/drive/1_8h8pZ5A0lb_MiyY5pOwc-F1UaY52IZZ)

Furthermore, let's explain the main functionality of our files.
* data_load     - in charge of loading the data from the json files
* preprocess    - contains main preprocessing functionality

**Issue tracking**<br/>
* clustering    - clustering and ranking _specifically_ for issue tracking
* extract_issue - Title extraction of issues and extraction of our 2 issues for detailed analysis
* main          - writing to and loading from .csv files for issues

**Event tracking**<br/>
* events        - all event related functionality, such as entity extraction, clustering, timeline, detailed information, etc.
* tfidf         - contains customized entity tokenizer for TfidfVectorizer

### Workflow
More detailed step-by-step can be found in the colab, but to sum up the workflow of our code

Issue-tracking: Load data -> preprocess -> cluster -> extract titles

Event-tracking: Pick 2 issues -> extract entities -> cluster -> group by 3 days -> timeline and detailed information


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

