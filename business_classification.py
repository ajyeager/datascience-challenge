from collections import defaultdict

def group_reviews(reviews, group_size=5):
    reviews_by_business = defaultdict(list)
    i = 0
    for review in reviews:
        business_id = review["business_id"]
        text = review["text"]
        if business_id in top_100:
            # tag each review with the corresponding index in the datset
            reviews_by_business[business_id].append((i, text))
        i += 1
    group_to_business = []
    for business in reviews_by_business:
        businesses = reviews_by_business[business]
        businesses_left = len(businesses)
        while(businesses_left >= group_size):
            group_to_business.append((businesses[:group_size], business))
            businesses = businesses[group_size:]
            businesses_left -= group_size
    return group_to_business



# most_common_items is the response from Counter(list).most_common(), a list consisting of (item, freq) pairs:
# if all items were chosen with equal frequency, then default to the label `fallback`
def get_most_common(most_common_items, fallback=None):
    # if the most frequent and least frequent items have the same frequency, fall back to the mode label
    if fallback is not None and most_common_items[0][1] == most_common_items[-1][1]:
        return fallback
    else:
        # if multiple items have the same top frequency, choose at random
        top = []
        topfreq = most_common_items[0][1]
        for (item, freq) in most_common_items:
            if freq < topfreq:
                break
            top.append(item)
        return np.random.choice(top)


def make_group_predictions(reviews_by_group, single_review_predictions, fallback=None):
    predicted = []
    actual = []
    for group in reviews_by_group:
        (group_reviews, actual_label) = group
        predicted_ids = []
        for (row_idx, _) in group_reviews:
            prediction = trim_label(single_review_predictions[row_idx])
            predicted_ids.append(prediction)
        top = get_most_common(Counter(predicted_ids).most_common(), fallback=fallback)
        predicted.append(top)
        actual.append(actual_label)
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    return predicted, actual