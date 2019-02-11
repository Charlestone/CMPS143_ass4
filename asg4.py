
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




DATA_DIR = "data"

random.seed(10)


MODEL_DIR = "models/"
OUTPUT_DIR = "output/"
FEATURES_DIR = "features/"





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


def main():

    all_feature_sets = [
        "word_pos_features", "word_features", "word_pos_liwc_features",
        #"word_embedding",
        #"liwc_features",
        #"binning_word_pos_features",
        #"binning_word_features", "binning_word_pos_liwc_features"
    ]



if __name__ == "__main__":
    main()










