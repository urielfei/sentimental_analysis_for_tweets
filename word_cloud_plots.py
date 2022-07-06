import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textual_data_cleaning import clean_tweets_df

depressive_words = ' '.join(list(clean_tweets_df[clean_tweets_df['label'] == 1]['clean_tweet']))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(depressive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

positive_words = ' '.join(list(clean_tweets_df[clean_tweets_df['label'] == 0]['clean_tweet']))
positive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(positive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(positive_wc)
plt.axis('off'),
plt.tight_layout(pad = 0)
plt.show()

