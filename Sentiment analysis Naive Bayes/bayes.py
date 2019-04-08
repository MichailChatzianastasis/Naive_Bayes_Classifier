
import csv
import pandas as pd

all_together = pd.read_csv("original.csv",encoding="ISO-8859-1")
test=a.loc[0:24999]
train=a.loc[25000:]

reviews=pd.Series.tolist(train.loc[:])
    
# Computing the prior (H=positive reviews) according to the Naive Bayes' equation

def get_H_count(score):
    # Compute the count of each classification occurring in the data
    return len([r for r in reviews if r[3] == score])

# We'll use these counts for smoothing when computing the prediction
positive_review_count = get_H_count('pos')
negative_review_count = get_H_count('neg')

# These are the prior probabilities (we saw them in the formula as P(H))
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)
print("P(H) or the prior is:", prob_positive)
# Python class that lets us count how many times items occur in a list
from collections import Counter
import re

def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone
    # Lowercase the text so that the algorithm doesn't see "Not" and "not" as different words, for example
    return " ".join([r[2].lower() for r in reviews if r[3] == score])

def count_text(text):
    # Split text into words based on whitespace -- simple but effective
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)

negative_text = get_text(reviews, "neg")
positive_text = get_text(reviews, "pos")

# Generate word counts(WC) dictionary for negative tone
negative_WC_dict = count_text(negative_text)

# Generate word counts(WC) dictionary for positive tone
positive_WC_dict = count_text(positive_text)
# H = positive review or negative review
def make_class_prediction(text, H_WC_dict, H_prob, H_count):
    prediction = 1
    text_WC_dict = count_text(text)
    
    for word in text_WC_dict:       
        prediction *=  text_WC_dict.get(word,0) * ((H_WC_dict.get(word, 0) + 1) / (sum(H_WC_dict.values()) + H_count))

        # Now we multiply by the probability of the class existing in the documents
    return prediction * H_prob

# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater
def make_decision(text):
    
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_WC_dict, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_WC_dict, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return -1
    return 1

print("For this review: {0}".format(reviews[0][0]))
print("")
print("The predicted label is ", make_decision(reviews[0][0]))
print("The actual label is ", reviews[0][1])

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0]) for r in test]
actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))