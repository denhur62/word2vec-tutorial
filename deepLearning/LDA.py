import pandas as pd
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", filename="abcnews-date-text.csv")
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

# 토큰화
data = data.head(50000)
text = data[['headline_text']]
text['headline_text'] = text.apply(
    lambda row: nltk.word_tokenize(row['headline_text']), axis=1)

# 불용어 제거
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(
    lambda x: [word for word in x if word not in (stop)])

# 표제어 추출
text['headline_text'] = text['headline_text'].apply(
    lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])

# 길이 3이하 자르기
tokenized_doc = text['headline_text'].apply(
    lambda x: [word for word in x if len(word) > 3])

# 역토큰화
detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['headline_text'] = detokenized_doc

# tf-idf
vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=1000)  # 상위 1,000개의 단어를 보존
X = vectorizer.fit_transform(text['headline_text'])

print(X.shape)

# 토픽 모델링LDA
lda_model = LatentDirichletAllocation(
    n_components=10, learning_method='online', random_state=777, max_iter=1)

lda_top = lda_model.fit_transform(X)
terms = vectorizer.get_feature_names()  # 단어 1천개


def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (
            idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])


get_topics(lda_model.components_, terms)
