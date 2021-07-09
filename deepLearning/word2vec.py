# import urllib.request
#import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)
print("---------- 토큰화 완료 ----------")
result = [word_tokenize(sentence) for sentence in normalized_text]
model = Word2Vec(sentences=result, vector_size=100, window=5,
                 min_count=5, workers=4, sg=0)
print("---------- word2Vec 완료 ----------")

model_result = model.wv.most_similar("man")
print(model_result)

model.wv.save_word2vec_format('eng_w2v')  # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")  # 모델 로드
model_result = loaded_model.most_similar("man")
print(model_result)
