import nltk
import string
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
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

def extract_nouns(text):
    # stopwords extension
    import calendar
    months = [x.lower() for x in calendar.month_name[1:13]]
    days = [x.lower() for x in calendar.day_name[0:7]]
    extensions = ['yonhap']
    stop_words = list(STOP_WORDS) + months + days + extensions

    tokens = nltk.word_tokenize(text)
    POS = nltk.pos_tag(tokens)
    lm = WordNetLemmatizer()

    result = ''
    res = []

    for word, pos in POS:
        select = ['NNP', 'NN']
        if pos in select:
            token = lm.lemmatize(word)
            token = token.lower()
            if token not in stop_words:
                res.append(token)
                result = ' '.join(res)

    result = remove_punctuation(result)

    return result


