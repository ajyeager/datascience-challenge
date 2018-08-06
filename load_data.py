import json
from collections import defaultdict
import sklearn

data_dir="/Users/vijay/Documents/code/yelp-dataset"
reviews_file = data_dir + "/yelp_academic_dataset_review.json"

reviews = []
reviews_raw = open(reviews_file).readlines()

for row in reviews_raw:
    review = json.loads(row)
    reviews.append(review)


def train_test_split(reviews):
    sklearn.model_selection.train_test_split(reviews)

for review in reviews:
    rating = review['stars']
    text = review['text']
    reviews_by_star = defaultdict(list)
