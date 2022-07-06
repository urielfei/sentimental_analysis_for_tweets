import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
raw_df = pd.read_csv('/Users/urielf/PycharmProjects/sentimental_analysis_for_tweets/sentiment_tweets3.csv')

names_dict = {'message to examine':'tweet','label (depression result)': 'label'}
tweets_df = raw_df.rename(names_dict,axis=1)
tweets_df = tweets_df[names_dict.values()]



def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2, clean=True):
    if clean:
        # specific
        message = re.sub(r"won't", "will not", message)
        message = re.sub(r"can\'t", "can not", message)
        message = re.sub(r"twitter", " ", message)

        # remove any url
        message = re.sub(r"http[s]?://\S+", " ", message)
        message = re.sub(r"quot", " ", message)
        message = re.sub(r"amp", " ", message)
        # general
        message = re.sub(r"n\'t", " not", message)
        message = re.sub(r"\'re", " are", message)
        message = re.sub(r"\'s", " is", message)
        message = re.sub(r"\'d", " would", message)
        message = re.sub(r"\'ll", " will", message)
        message = re.sub(r"\'t", " not", message)
        message = re.sub(r"\'ve", " have", message)
        message = re.sub(r"\'m", " am", message)

        # special char
        message = re.sub(r'[^A-Za-z0-9]+', " ", message)

        # message = re.sub(r"@+"," ", message)

        # remove any thing with html tags
        # message = cleanhtml(message)
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if clean:
        words = [re.sub(r'https?://\S+', '', word) for word in words]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

def get_clean_tweet(df):
    result = dict()
    for (i, message) in enumerate(df['tweet']):
        processed_message = process_message(message,gram=1)
        result[i] = " ".join(processed_message)
    df['clean_tweet'] = result.values()
    return df



clean_tweets_df = get_clean_tweet(tweets_df)


