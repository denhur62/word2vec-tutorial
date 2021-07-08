import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=1000,
                             max_df=0.5,
                             smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])

svd_model = TruncatedSVD(
    n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)

terms = vectorizer.get_feature_names()


def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (
            idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])


get_topics(svd_model.components_, terms)
