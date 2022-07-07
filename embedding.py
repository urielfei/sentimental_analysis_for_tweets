import pandas as pd
from sklearn.model_selection import train_test_split
from textual_data_cleaning import clean_tweets_df
from sklearn.feature_extraction.text import TfidfVectorizer
import time
start = time.time()

def embed_tf_idf(df):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df['clean_tweet'])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    embeddings = pd.DataFrame(denselist, columns=feature_names)
    return embeddings

embeddings = embed_tf_idf(clean_tweets_df)
labels = clean_tweets_df['label']

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels,
                                          test_size=0.2, random_state=42)
print ("Training set shapes:", X_train.shape, y_train.shape)
print ("Test set shapes:", X_test.shape, y_test.shape)

print(X_train.columns)
print(time.time()-start)