from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())  # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_)  # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
