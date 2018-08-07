from collections import defaultdict
import load_data
import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = "/Users/vijay/Documents/code/yelp-dataset"
reviews_file = data_dir + "/yelp_academic_dataset_review.json"

start = time.time()

reviews = []
reviews_raw = open(reviews_file).readlines()

print("0: elapsed time: %s" % (time.time() - start))

for row in reviews_raw:
    review = json.loads(row)
    reviews.append(review)

print("1: elapsed time: %s" % (time.time() - start))


train, test = load_data.train_test_split(reviews)
fasttext_data_dir = "/Users/vijay/Documents/code/datascience-challenge"
load_data.preprocess_for_fasttext(train, fasttext_data_dir + "/train.fasttext")
load_data.preprocess_for_fasttext(test, fasttext_data_dir + "/test.fasttext")

print("elapsed time: %s" % (time.time() - start))

[train_data, test_data] = [load_data.reviews_by_star(data) for data in [train, test]]


print("2: " + "\n".join(["%s:%s" % (k, len(v)) for (k, v) in reviews_by_rating.items()]))

# TODO(Vijay): add ML models to predict the rating from each movie


print("3: elapsed time: %s" % (time.time() - start))


def tfidf_feature_extraction(data):
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=10000, lowercase=True)
    tfidf.fit(data)
    print("incomplete")