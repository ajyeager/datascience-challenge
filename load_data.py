import json
from collections import defaultdict
from sklearn import model_selection

data_dir="/Users/vijay/Documents/code/yelp-dataset"
reviews_file = data_dir + "/yelp_academic_dataset_review.json"

reviews = []
reviews_raw = open(reviews_file).readlines()

for row in reviews_raw:
    review = json.loads(row)
    reviews.append(review)

train, test = train_test_split(reviews)

reviews_by_rating = reviews_by_star(train)

# TODO(Vijay): add ML models to predict the rating from each movie

def train_test_split(reviews):
    [train, test] = model_selection.train_test_split(reviews, test_size=0.25)
    return train, test

def reviews_by_star(train):
    reviews_by_star = defaultdict(list)
    for review in train:
        rating = review['stars']
        text = review['text']
        reviews_by_star[rating].append(text)
    return reviews_by_star