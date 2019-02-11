import re, nltk, pickle, argparse
import os, sys
import data_helper
from features import get_features_category_tuples


from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import sklearn, numpy
import pandas as pd
import random
from sklearn import tree
from nltk.classify import SklearnClassifier

def build_classifier(classifier_name):
    """
    Accepted names: nb, dt, svm, sk_nb, sk_dt, sk_svm

    svm and sk_svm will return the same type of classifier.

    :param classifier_name:
    :return:
    """
    if classifier_name == "nb":
        cls = nltk.classify.NaiveBayesClassifier
    elif classifier_name == "nb_sk":
        cls = SklearnClassifier(BernoulliNB())
    elif classifier_name == "dt":
        cls = nltk.classify.DecisionTreeClassifier
    elif classifier_name == "dt_sk":
        cls = SklearnClassifier(tree.DecisionTreeClassifier())
    elif classifier_name == "svm_sk" or classifier_name == "svm":
        cls = SklearnClassifier(svm.SVC())
    else:
        assert False, "unknown classifier name:{}; known names: nb, dt, svm, nb_sk, dt_sk, svm_sk".format(classifier_name)

    return cls

def train(classifier, dataset):

def main():
    parser = argparse.ArgumentParser(description='Assignment 4.1.1')
    parser.add_argument('-tr', dest="train_data", default="train_examples.tsv",
                        help='File name of the training data.')
    parser.add_argument('-d', dest="dev_data", default="dev_examples.tsv",
                        help='File name of the development data.')
    parser.add_argument('-t', dest="test_data", default="test_examples",
                        help='File name of the test data')
    parser.add_argument('-w', dest="write_fname", default=None,
                        help='File name of the output.')
    parser.add_argument('-b', dest="binning", default=False,
                        help='Whether you want to bin the features or not')
    parser.add_argument('-c', dest="classifier", default=None,
                        help='Classifier already trained')
    args = parser.parse_args()

    # train a NB classifier
    nb = build_classifier("nb")
    nb = train(nb, args.train_data)

if __name__ == "__main__":
    main()