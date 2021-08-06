import re
import numpy as np
from scipy.sparse.dia import dia_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def partial_fit(self, X):
    max_idx = max(self.vocabulary_.values())
    for a in X:
        # update vocabulary_
        if self.lowercase:
            a = a.lower()
        tokens = re.findall(self.token_pattern, a)
        for w in tokens:
            if w not in self.vocabulary_:
                max_idx += 1
                self.vocabulary_[w] = max_idx

        # update idf_
        df = (self.n_docs + self.smooth_idf) / \
            np.exp(self.idf_ - 1) - self.smooth_idf
        self.n_docs += 1
        df.resize(len(self.vocabulary_))
        for w in tokens:
            df[self.vocabulary_[w]] += 1
        idf = np.log((self.n_docs + self.smooth_idf) /
                     (df + self.smooth_idf)) + 1
        self._tfidf._idf_diag = dia_matrix(
            (idf, 0), shape=(len(idf), len(idf)))


TfidfVectorizer.partial_fit = partial_fit
articleList = ['안녕하세요',
               '반갑습니다.', '환영 합니다. ']
vec = TfidfVectorizer()
vec.fit_transform(articleList)
print(vec.shape)
vec.n_docs = len(articleList)
vec.partial_fit(['추가가 가능할까요'])

example = "추가 좋습니다"

example_vector = vec.transform([example])
print(vec.transform([articleList]))
print(example_vector)

cosine_similar = linear_kernel(
    example_vector, vec.transform([articleList])).flatten()
ex = cosine_similar.argsor()[::-1]
print(ex)
