from collections import defaultdict
from sklearn import model_selection
import math
import string


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


def preprocess(text, return_tokens=False):
    # make text all lowercase
    text = text.lower()
    # remove punctuation

    punctuation_allowed = ["'", '-', ':', '$']
    for p in string.punctuation:
        if p in punctuation_allowed:
            continue
        text = text.replace(p, ' ')
    # replace all whitespace with " " (single-space)
    tokens = text.split()
    #the punctuation "-", ":", "$" are allowed, as long as they don't live at the end of a token
    tokens = [t for t in tokens if t[-1] not in ['-', ':', '$']]
    if return_tokens:
        return tokens
    else:
        text = " ".join(tokens)
    return text

# given a datset of reviews, produce data in the fasttext format for a given label field
def preprocess_for_fasttext(dataset, outname, label_key = 'stars'):
    outfile = open(outname, 'w')
    lines = 0
    for review in dataset:
        label = review[label_key]
        text = preprocess(review['text'])
        # fast text expects the label formatted as __label__1 for label 1
        annotated_line = "__label__%s %s\n" % (label, text)
        outfile.write(annotated_line)
        lines += 1
    print("%s lines written to %s" % (lines, outname))
    outfile.close()

def trim_label(labelstr, prefix="__label__"):
    remove_len = len(prefix)
    label = labelstr[remove_len:]
    return label.strip()