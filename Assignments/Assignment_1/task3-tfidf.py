import datetime
import csv

from pprint import pprint

from gensim import models
from gensim import similarities
from gensim import corpora

from nltk.corpus import stopwords
from nltk import PorterStemmer

with open("news2.csv") as f:
    reader = csv.DictReader(f)
    articles = []

    for row in reader:
        articles.append(row)

docs = [row['article_section'] + ',' + row['description'] for row in articles]

init_t = datetime.datetime.now()

stemmer = PorterStemmer()
stoplist = stopwords.words('english')

txts = [
    [stemmer.stem(word) for word in _doc.lower().split() if word not in stoplist]
    for _doc in docs
]

mapping = corpora.Dictionary(txts)
model_bow = [mapping.doc2bow(txt) for txt in txts]

tfidf = models.TfidfModel(model_bow)
tfidf_v = tfidf[model_bow]

id2token = dict(mapping.items())


def convert(match):
    return mapping.id2token[int(match.group(0)[0:-1])]


tfidf_msim = similarities.MatrixSimilarity(tfidf_v)

model_create_t = datetime.datetime.now()
elapsed_time_model_creation: datetime = model_create_t - init_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')

total_goods = 0
total_sports_docs = 0
for doc in docs:
    if doc.split(',')[0] != "Sports":
        continue
    total_sports_docs += 1

    doc_stemmed = [stemmer.stem(word) for word in doc.lower().split() if word not in stoplist]
    vec_bow = mapping.doc2bow(doc_stemmed)
    doc_tfidf_v = tfidf[vec_bow]

    sims = tfidf_msim[doc_tfidf_v]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    _i = 0
    for doc_position, doc_score in sims:
        if _i == 0:
            _i += 1
            continue
        if _i > 10:
            break
        if docs[doc_position][:6] == "Sports":
            total_goods += 1
        _i += 1

print("Ratio quality: " + str(total_goods / (total_sports_docs * 10)))

end_t: datetime = datetime.datetime.now()
elapsed_time_comparison: datetime = end_t - model_create_t
print()
print('Execution time comparison:', elapsed_time_comparison, 'seconds')