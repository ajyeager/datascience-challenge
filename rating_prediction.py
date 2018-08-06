from collections import defaultdict
import load_data
import json

data_dir = "/Users/vijay/Documents/code/yelp-dataset"
reviews_file = data_dir + "/yelp_academic_dataset_review.json"

reviews = []
reviews_raw = open(reviews_file).readlines()

for row in reviews_raw:
    review = json.loads(row)
    reviews.append(review)

train, test = load_data.train_test_split(reviews)
reviews_by_rating = load_data.reviews_by_star(train)

# TODO(Vijay): add ML models to predict the rating from each movie
