import os

# e.g. reviews_fname = "/Users/vijay/Documents/vijay/yelp-dataset/yelp_academic_dataset_review.json"
#      out_dir = "/Users/vijay/Documents/vijay/yelp-dataset/task2"
def group_reviews_by_restaurant(reviews_fname, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


'''
featurizer='bow'
fpath = '/Users/vijay/Desktop/train_test_features_and_labels_%s.p' % featurizer
try:
    pickle.dump([train_features, train_labels, test_features, test_labels], open(fpath, 'wb'))
    print("wrote features to %s (using wb)" % fpath)
except Exception as e:
    pickle.dump([train_features, train_labels, test_features, test_labels], open(fpath, 'w'))
    print("wrote features to %s (using w)" % fpath)

pickle.dump((p, tfidf), open('train_featuredetector', 'wb'))
'''



def batch_transform_and_train():
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000, lowercase=True)
    batch_size = 100000
    data_seen = 0
    while data_seen <= len(train):
        batch = [row['text'] for row in train[data_seen:data_seen+batch_size]]
        vectorizer.fit(batch)
    clf = SGDRegressor()
    data_seen = 0
    while data_seen <= len(train):
        train_text = [row['text'] for row in train[data_seen:data_seen+batch_size]]
        train_labels = [row['stars'] for row in train[data_seen:data_seen+batch_size]]
        learn_batch(clf, train_text, train_labels, lambda data: vectorizer.transform(data))
    return vectorizer, clf