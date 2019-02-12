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

def feat_selection(classifier, train_dataset, dev_dataset, test_dataset, binning):
    best = (0.0, 0)
    best_features = classifier.most_informative_features(1000000)
    selected_features = None
    for i in [2 ** i for i in range(5, 20)]:
        selected_features = set([fname for fname, value in best_features[:i]])
        fct_train, texts_train = build_features(train_dataset, "word_features", binning)
        for feat_dict, label in fct_train:
            feat_dict = select_features(feat_dict, selected_features)
        classifier = nltk.NaiveBayesClassifier.train(fct_train)
        fct_dev, texts_dev = build_features(dev_dataset, "word_features", binning)
        for feat_dict, label in fct_dev:
            feat_dict = select_features(feat_dict, selected_features)
        accuracy, cm = select_evaluate(classifier, fct_dev)
        print("{0:6d} {1:8.5f}".format(i, accuracy))
        print_results(write_fname, 'development', i, accuracy, cm)
        fct_test, texts_test = build_features(test_dataset, "word_features", binning)
        accuracy, cm = select_evaluate(classifier, fct_test)
        print_results(write_fname, 'test', i, accuracy, cm)
        if accuracy > best[0]:
            best = (accuracy, i)

    return set([fname for fname, value in best_features[:best[1]]]), best

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

    # 1.1 and 1.2
    nb = build_classifier("nb")
    sys.stdout = open(write_fname, 'a+')
    print("Training different Naïve Bayes classifiers: ")
    sys.stdout = sys.__stdout__
    print("Training the model...")
    nb = train(nb, args.train_data, args.binning)
    print("Evaluating the model on the development set...")
    dev_acc, dev_cm = evaluate(nb, args.dev_data, args.binning)
    print_results(write_fname, 'development', 'All', dev_acc, dev_cm)
    print("Evaluating the model on the test set...")
    test_acc, test_cm = evaluate(nb, args.test_data, args.binning)
    print_results(write_fname, 'test', 'All', test_acc, test_cm)
    print("Selecting best set of features...")
    selected_features, best = feat_selection(nb, args.train_data, args.dev_data, args.test_data, args.binning)
    sys.stdout = open(args.write_fname, 'a+')
    print('Best set of features: {}'.format(best[1]))
    sys.stdout = sys.__stdout__
    print("Predicting with the best NB classifier...")
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
    print_results(write_fname, 'test', best[1], accuracy, cm)
    sys.stdout = open("nb-word_features-{}_features.txt".format(best[1]), 'a+')
    print('Predictions of the best NB classifier.:')
    print(best_nb.classify_many([feat[0] for feat in fct_test]))
    sys.stdout = sys.__stdout__
    # 2.1
    nb_sk = build_classifier("nb_sk")
    print("Training BernoulliNB classifier...")
    sys.stdout = open(write_fname, 'a+')
    print("Training BernoulliNB classifier: ")
    sys.stdout = sys.__stdout__
    nb_sk = nb_sk.train(fct_train)
    accuracy, cm = select_evaluate(nb_sk, fct_dev)
    print_results(write_fname, 'dev', best[1], accuracy, cm)
    accuracy, cm = select_evaluate(nb_sk, fct_test)
    print_results(write_fname, 'test', best[1], accuracy, cm)
    dt_sk = build_classifier("dt_sk")
    print("Training Decision Tree classifier...")
    sys.stdout = open(write_fname, 'a+')
    print("Training Decision Tree classifier: ")
    sys.stdout = sys.__stdout__
    dt_sk = dt_sk.train(fct_train)
    accuracy, cm = select_evaluate(dt_sk, fct_dev)
    print_results(write_fname, 'dev', best[1], accuracy, cm)
    accuracy, cm = select_evaluate(dt_sk, fct_test)
    print_results(write_fname, 'test', best[1], accuracy, cm)
    # 2.2.1
    print("Generating w2vec features...")
    train_w2vec_feats = build_w2vec(args.train_data)
    dev_w2vec_feats = build_w2vec(args.dev_data)
    test_w2vec_feats = build_w2vec(args.test_data)
    # 2.2.2
    svm = build_classifier("svm")
    print("Training SVM classifier with word features...")
    sys.stdout = open(write_fname, 'a+')
    print("Training SVM classifier with word features: ")
    sys.stdout = sys.__stdout__
    svm = svm.train(fct_train)
    accuracy, cm = select_evaluate(svm, fct_dev)
    print_results(write_fname, 'dev', best[1], accuracy, cm)
    accuracy, cm = select_evaluate(svm, fct_test)
    print_results(write_fname, 'test', best[1], accuracy, cm)
    svm_w2v = build_classifier("svm")
    print("Training SVM classifier with w2vec features...")
    sys.stdout = open(write_fname, 'a+')
    print("Training SVM classifier with w2vec features: ")
    sys.stdout = sys.__stdout__
    svm_w2v = svm_w2v.train(train_w2vec_feats)
    accuracy, cm = select_evaluate(svm_w2v, dev_w2vec_feats)
    print_results(write_fname, 'dev', 'All', accuracy, cm)
    accuracy, cm = select_evaluate(svm_w2v, test_w2vec_feats)
    print_results(write_fname, 'test', 'All', accuracy, cm)


if __name__ == "__main__":
    main()