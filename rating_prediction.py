from collections import defaultdict
import load_data
import json
import time
import math
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDRegressor
from pymagnitude import Magnitude
vectors = Magnitude("/Users/vijay/Documents/vijay/datascience-challenge/model.magnitude")


def prepare_fast_text_data(fasttext_data_dir, train_data, test_data):
    train_out = fasttext_data_dir + "/train.fasttext"
    load_data.preprocess_for_fasttext(train_data, train_out)
    test_out = fasttext_data_dir + "/test.fasttext"
    load_data.preprocess_for_fasttext(test_data, test_out)
    test_labels_out = fasttext_data_dir + "/test.fasttext.label_only"
    test_labels_only = open(test_labels_out)
    for review in test_data:
        label = review['stars']
        # fast text expects the label formatted as __label__1 for label 1
        annotated_line = "__label__%s %s\n" % (label, text)
        test_labels_only.write(annotated_line)
        lines += 1
    test_labels_only.close()
    print("wrote out fasttext data files to %s, %s, and %s" % (train_out, test_out, test_labels_out))

def run_fast_text_model():
    print("run ../fastText-0.1.0/fasttext predict model.bin test.fasttext > test_predicted_fasttext3")



def rmse(predicted, actual):
    if np.shape(predicted) != np.shape(actual):
        raise ValueError("predicted and actual must have same length to compute rmse - %s != %s" %
                         (len(predicted), len(actual)))
    rmse_sum = 0.0
    for i in range(len(predicted)):
        rmse_sum += math.pow(predicted[i] - actual[i], 2)
    return math.sqrt(rmse_sum / len(predicted))

def classification_accuracy(predicted, actual):
    if np.shape(predicted) != np.shape(actual):
        raise ValueError("predicted and actual must have same length to compute rmse - %s != %s" %
                         (len(predicted), len(actual)))
    matches = np.asarray(predicted) == np.asarray(actual)
    return sum(matches) / len(matches)

def evaluate_fasttext_predictions(fasttext_predictions_fname, test_labels_fname, label_prefix="__label__", label_key='stars'):
    remove_len = len(label_prefix)
    predicted_labels = []
    count = 0
    for label_row in open(fasttext_predictions_fname):
        predicted_labels.append(int(label_row[remove_len:]))
    true_labels = []
    for test_label_row in open(test_labels_fname):
        true_labels.append(int(test_label_row[remove_len:]))
    return classification_accuracy(predicted_labels, true_labels), rmse(predicted_labels, true_labels)



'''
compute_word_embeddings:
reviews: set of text reviews
outdir: path to a directory to write out embedding files
prefix: label for embedding file name (e.g. "test_features"/"test_labels")
offset: number of lines to skip before writing vectors (for resuming partial runs)

'''
def compute_word_embeddings(reviews, outdir, data_prefix = "nolabel", label_key = 'stars', embedding_size = 100, batch_write_size = 100000, offset = 0):
    if not os.path.exists(os.path.dirname(outdir)):
        os.mkdir(outdir)
    feature_fname = outdir + "/%s_features" % data_prefix
    labels_fname = outdir + "/%s_labels" % data_prefix
    features_file = open(feature_fname, 'ab')
    labels_file = open(labels_fname, 'a')
    total_preprocessing_time, total_featurization_time = 0, 0
    counter = 0
    labels_batch, text_batch = [], []
    seconds_counter = time.time()
    for review in reviews:
        counter += 1
        if offset > 0 and counter <= offset:
            continue
        s = time.time()
        text = load_data.preprocess(review['text'], return_tokens=True)
        stars = review[label_key]
        total_preprocessing_time += time.time() - s
        #
        if len(text) == 0:
            text = [None]
            stars = -1
        text_batch.append(text)
        labels_batch.append(stars)
        if counter % 1000 == 0:
            print("%s out of %s reviews processed in %s seconds - totals so far - total preprocessing time: %s, total featurization time so far: %s" % (counter, len(reviews), time.time() - seconds_counter, total_preprocessing_time, total_featurization_time))
            seconds_counter = time.time()
        if counter % batch_write_size == 0 or counter == len(reviews):
            s = time.time()
            try:
                w2v = np.average(vectors.query(text_batch), axis=1)
            except Exception as e:
                print("produced empty w2v at batch %s" % counter)
                print("error:", e)
                w2v = np.empty(embedding_size)
                w2v[:] = 0
            total_featurization_time += time.time() - s
            #
            np.savetxt(features_file, w2v, delimiter= ' ')
            features_file.flush()
            #
            labels_file.write("\n".join([str(r) for r in labels_batch]) + "\n")
            labels_file.flush()
            #
            text_batch = []
            labels_batch = []
            print("wrote %s rows to %s and %s" % (counter, feature_fname, labels_fname))
        #
    features_file.close()
    labels_file.close()
    print("wrote files to %s and %s" % (features_file, labels_file))
    print("took %s seconds for preprocessing" % total_preprocessing_time)
    print("took %s seconds for feature extraction" % total_featurization_time)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def round_regressor_predictions(continuous_predictions, discrete_choices = [1,2,3,4,5]):
    predictions = []
    for pred in continuous_predictions:
        predictions.append(find_nearest(discrete_choices, pred))
    return np.asarray(predictions)


'''
BOW/tf-idf feature extraction code (excluded from final report):
'''

def extract_features(dataset, label_key = 'stars', featurizer='tfidf'):
    text, labels = [], []
    for row in dataset:
        text.append(load_data.preprocess(row['text']))
        labels.append(row[label_key])
    if featurizer == 'tfidf':
        features = tfidf_feature_extraction(text)
    else:
        features = bow_feature_extraction(text)
    return features, labels


def bow_feature_extraction(data):
    tfidf = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000, lowercase=True)
    return tfidf.fit_transform(data)


def tfidf_feature_extraction(data):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=50000, lowercase=True)
    return tfidf.fit_transform(data)

# compute features in batches to avoid eating all available memory
def learn_batch(clf, features, labels, vectorize):
    feats = vectorize(features)
    clf.fit(feats, labels)
    return clf

# train an online-friendly SGDRegressor in batches
def batch_transform_and_train():
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000, lowercase=True)
    batch_size = 100000
    data_seen = 0
    while data_seen <= len(train):
        batch = [row['text'] for row in train[data_seen:data_seen+batch_size]]
        vectorizer.fit(batch)
        print("%s: elapsed time: %s" % (data_seen, time.time() - start))
    clf = SGDRegressor()
    data_seen = 0
    while data_seen <= len(train):
        train_text = [row['text'] for row in train[data_seen:data_seen+batch_size]]
        train_labels = [row['stars'] for row in train[data_seen:data_seen+batch_size]]
        learn_batch(clf, train_text, train_labels, lambda data: vectorizer.transform(data))
    return vectorizer, clf