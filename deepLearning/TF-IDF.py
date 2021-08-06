from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus = ["휴일인 오늘도 서쪽을 중심으로 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.",
          "폭염을 피해서 휴일에 놀러왔다가 갑작스런 비로 인해 망연자실하고 있습니다.", "안녕하세요"]
# vector = CountVectorizer()
# print(vector.fit_transform(corpus).toarray())  # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
# print(vector.vocabulary_)  # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

tfidfv = TfidfVectorizer()
tfidfv.fit(corpus[:2])
print("1", tfidfv.vocabulary_)
# print(tfidfv.transform(corpus).toarray())
# print(tfidfv.vocabulary_)
tfidfv.fit_transform(corpus)
print("2", tfidfv.vocabulary_)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)  # 문장 벡터화 진행
idf = tfidf_vectorizer.idf_
#print(dict(zip(tfidf_vectorizer.get_feature_names(), idf)))
#print(float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])))
