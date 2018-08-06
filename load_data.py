from collections import defaultdict
from sklearn import model_selection


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
