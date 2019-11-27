import nltk
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(df):
    df['edited'] = df['body']
    # df['edited'] = df['description']
    # ~30 seconds
    df['edited'] = df['edited'].map(lambda x: normalize(x))
    # ~30 seconds
    df['edited'] = df['edited'].map(lambda x: nltk.word_tokenize(x))
    # ~ 30 seconds
    df['edited'] = df['edited'].map(lambda x: lemmatization(x))

    df['edited'] = df['edited'].map(lambda x: ' '.join(word for word in x))

    return df

def normalize(text):
    text = text.lower()
    text = text.strip()
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = remove_short_words(text)

    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    new_string = []
    for word in text.split():
        if word not in stopwords:
            new_string.append(word)
    return ' '.join(new_string)


def remove_punctuation(text):
    for char in ["‘", "’"]:  # Weird ' from text
        text = text.replace(char, "'")
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_short_words(text):
    return ' '.join(word for word in text.split() if len(word) > 3)

def lemmatization(words):
    lm = WordNetLemmatizer()
    return [lm.lemmatize(word) for word in words]

# ===== Not used in preprocess =====
def stemming(words):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]


