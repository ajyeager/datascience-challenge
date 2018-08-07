from collections import defaultdict
from sklearn import model_selection
import math


def train_test_split(reviews):
    [train, test] = model_selection.train_test_split(reviews, test_size=0.25)
    return train, test


def reviews_by_star(train):
    reviews_by_rating = defaultdict(list)
    for review in train:
        rating = review['stars']
        text = review['text']
        reviews_by_rating[rating].append(text)
    return reviews_by_rating


def preprocess(text):
    text = text.lower()

def preprocess_for_fasttext(dataset, outname):
    outfile = open(outname, 'w')
    lines = 0
    for review in dataset:
        rating = review['stars']
        text = preprocess(review['text'])
        # fast text expects the label formatted as __label__1 for label 1
        annotated_line = "__label__%s %s\n" % (rating, text)
        outfile.write(annotated_line)
        lines += 1
    print("%s lines written to %s" % (lines, outname))
    outfile.close()

def evaluate_test(test, predictor):
    predictions = []
    rmse = 0.0
    for review in test:
        predicted = predictor(review['text'])
        rmse += math.pow(predicted - review['stars'], 2)