import pandas as pd
from sklearn.datasets import fetch_20newsgroups
categories = [
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print(newsgroups_train.data[1])
