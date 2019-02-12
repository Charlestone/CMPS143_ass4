import re, nltk, pickle, argparse
import os, sys
import data_helper
from features import get_features_category_tuples, get_w2v


from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import sklearn, numpy
import pandas as pd
import random
from sklearn import tree
from nltk.classify import SklearnClassifier

write_fname = None

def bin(count):
    """
    Results in bins of  0, 1, 2, 3 >=
    :param count: [int] the bin label
    :return:
    """
    the_bin = None
    ###     YOUR CODE GOES HERE
    the_bin = count if count < 3 else 3

    return the_bin

def build_features(data_file, feat_name, binning):

    # read text data
    category_texts = data_helper.get_reviews(os.path.join("./", data_file))

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name, binning)

    return features_category_tuples, texts

def build_w2vec(data_file):
    # read text data
    category_texts = data_helper.get_reviews(os.path.join("./", data_file))
    w2vec_feats = get_w2v(category_texts)
    return w2vec_feats

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

def train(classifier, dataset, binning):

    feature_category_tuples, texts = build_features(dataset, "word_features", binning)
    return classifier.train(feature_category_tuples)

def select_train(classifier, tuples):
    return classifier.train(tuples)

def evaluate(classifier, dataset, binning):

    feature_category_tuples, texts = build_features(dataset, "word_features", binning)
    features = [feat[0] for feat in feature_category_tuples]
    real_labels = [feat[1] for feat in feature_category_tuples]
    accuracy = nltk.classify.accuracy(classifier, feature_category_tuples)
    predicted_labels = classifier.classify_many(features)
    confusion_matrix = nltk.ConfusionMatrix(real_labels, predicted_labels)
    return accuracy, confusion_matrix

def select_evaluate(classifier, tuples):
    features = [feat[0] for feat in tuples]
    real_labels = [feat[1] for feat in tuples]
    accuracy = nltk.classify.accuracy(classifier, tuples)
    predicted_labels = classifier.classify_many(features)
    confusion_matrix = nltk.ConfusionMatrix(real_labels, predicted_labels)
    return accuracy, confusion_matrix

def print_results(file, dataset_name, feat_num, acc, cm):
    if file is not None:
        sys.stdout = open(file, 'a+')
        print('{} {} features info: \nAccuracy: {} \n{}'.format(dataset_name, feat_num, acc, cm))
        sys.stdout = sys.__stdout__

def select_features(dict, selected_features):
    pop_list = []
    for key in dict.keys():
        if key not in selected_features:
            pop_list.append(key)
    for key in pop_list:
        dict.pop(key)
    return dict

def main():
    parser = argparse.ArgumentParser(description='Assignment 4')
    parser.add_argument('-tr', dest="train_data", default="train_examples.tsv",
                        help='File name of the training data.')
    parser.add_argument('-d', dest="dev_data", default="dev_examples.tsv",
                        help='File name of the development data.')
    parser.add_argument('-t', dest="test_data", default="test_examples.tsv",
                        help='File name of the test data')
    parser.add_argument('-w', dest="write_fname", default=None,
                        help='File name of the output.')
    parser.add_argument('-b', dest="binning", default=False,
                        help='Whether you want to bin the features or not')
    parser.add_argument('-c', dest="classifier", default=None,
                        help='Classifier already trained')
    args = parser.parse_args()
    global write_fname
    write_fname = args.write_fname
    nb = build_classifier("nb")
    nb = train(nb, args.train_data, args.binning)
    selected_features = set([fname for fname, value in nb.most_informative_features(100000)[:32768]])
    best_nb = build_classifier("nb")
    fct_train, texts_train = build_features(args.train_data, "word_features", args.binning)
    for feat_dict, label in fct_train:
        feat_dict = select_features(feat_dict, selected_features)
    best_nb = best_nb.train(fct_train)
    fct_dev, texts_dev = build_features(args.dev_data, "word_features", args.binning)
    for feat_dict, label in fct_dev:
        feat_dict = select_features(feat_dict, selected_features)
    fct_test, texts_test = build_features(args.test_data, "word_features", args.binning)
    for feat_dict, label in fct_test:
        feat_dict = select_features(feat_dict, selected_features)
    accuracy, cm = select_evaluate(best_nb, fct_test)
    print_results(write_fname, 'test', '32768', accuracy, cm)
    sys.stdout = open("nb-word_features-{}_features.txt".format(best[1]), 'a+')
    print('Predictions of the best NB classifier.:')
    print(best_nb.classify_many([feat[0] for feat in fct_test]))
    sys.stdout = sys.__stdout__
if __name__ == "__main__":
    main()