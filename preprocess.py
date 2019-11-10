from data_load import load
import nltk
# nltk.download('stopwords') RUN ONCE
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
# nltk.download('wordnet')


def normalize(text):
    text = text.lower()
    text = text.strip()
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = remove_short_words(text)

    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    for word in stopwords:
        text.replace(word, '')
    return text


def remove_punctuation(text):
    for char in ["‘", "’"]:  # Weird ' from text
        text = text.replace(char, "'")
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_short_words(text):
    return ' '.join(word for word in text.split() if len(word) > 3)

def stemming(words):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

def lemmatization (words):
    lm = WordNetLemmatizer()
    return [lm.lemmatize(word) for word in words]


df = load('articles/')
df['edited'] = df['body']

# ~30 seconds
df['edited'] = df['edited'].map(lambda x: normalize(x))

# ~30 seconds
df['edited'] = df['edited'].map(lambda x: nltk.word_tokenize(x))

# TODO: Pick either stemming or lemmatization
# ~150 seconds
df['edited'] = df['edited'].map(lambda x: stemming(x))
# ~ 30 seconds
df['edited'] = df['edited'].map(lambda x: lemmatization(x))


print(df['edited'][0])
