from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)

# in NLTK
text = ["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words=sw)
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
[[1 1 1 1]]
{'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
