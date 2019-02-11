
import re, nltk, pickle, argparse
import os
import data_helper
import sys
from features import get_features_category_tuples

DATA_DIR = "data"


def write_features_category(features_category_tuples, output_file_name, test):
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    if test:
        for (features, category) in features_category_tuples:
            output_file.write("{0}\n".format(features))
    else:
            for (features, category) in features_category_tuples:
                output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):

    ###     YOUR CODE GOES HERE
    # TODO: evaluate your model
    features = [feat[0] for feat in features_category_tuples]
    real_labels = [feat[1] for feat in features_category_tuples]
    accuracy = nltk.classify.accuracy(classifier, [(features[i], real_labels[i]) for i in range(len(features))])
    predicted_labels = classifier.classify_many(features)
    confusion_matrix = nltk.ConfusionMatrix(real_labels, predicted_labels)
    return accuracy, confusion_matrix


def build_features(data_file, feat_name, binning, save_feats=None, test=False):

    # read text data
    positive_texts, negative_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))
    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name, binning)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats, test)

    return features_category_tuples, texts



def train_model(datafile, feature_set, binning, save_model=None):

    features_data, texts = build_features(datafile, feature_set, binning,
                                              feature_set+'-'+'training')

    ###     YOUR CODE GOES HERE
    # TODO: train your model here
    classifier = nltk.NaiveBayesClassifier.train(features_data)


    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier


def train_eval(train_file, feature_set, binning, eval_file=None, classifier_fname=None):

    # train the model
    split_name = "train"

    # save the model
    if classifier_fname is not None:
        model = get_classifier(classifier_fname)
    else:
        # train the model
        split_name = "train"
        model = train_model(train_file, feature_set, binning)

    sys.stdout = open(feature_set + '-informative-features.txt', 'w')
    print(model.show_most_informative_features(20))
    sys.stdout = sys.__stdout__
    # evaluate the model
    if eval_file is not None:
        features_data, texts = build_features(eval_file, feature_set, binning,
                                              feature_set+'-'+'development')
        accuracy, cm = evaluate(model, features_data, texts, data_set_name=None)
        if feature_set is "word_features":
            sys.stdout = open('output-ngrams.txt', 'w')
        elif feature_set is "word_pos_features":
            sys.stdout = open('output-pos.txt', 'w')
        else:
            sys.stdout = open('output-liwc.txt', 'w')

        print("The accuracy of {} is: {}".format(eval_file, accuracy))
        print("Confusion Matrix:")
        print(str(cm))

        sys.stdout = sys.__stdout__
    else:
        accuracy = None

    return model


def main():


    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('-d', dest="data_fname", default="train_examples.tsv",
                        help='File name of the training data.')
    parser.add_argument('-t', dest="test_data", default=None,
                        help='File name of the test data')
    parser.add_argument('-w', dest="write_fname", default=None,
                        help='File name of the output.')
    parser.add_argument('-b', dest="binning", default=False,
                        help='Whether you want to bin the features or not')
    parser.add_argument('-c', dest="classifier", default=None,
                        help='Classifier already trained')
    args = parser.parse_args()


    train_data = args.data_fname


    eval_data = "dev_examples.tsv"


    best_classifier = None
    best_feat_set = []
    best_conf = None
    best_acc = 0
    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features"]:
        print("\nTraining with {}".format(feat_set))
        classifier = train_eval(train_data, feat_set, args.binning, eval_data, args.classifier)
        features, texts = build_features(eval_data, feat_set, args.binning)
        acc, confusion = evaluate(classifier, features, texts)
        if acc > best_acc:
            best_classifier = classifier
            best_feat_set = feat_set
            best_acc = acc
            best_conf = confusion

    sys.stdout = open('output-best.txt', 'w')
    print("The accuracy of {} is: {}".format(feat_set, best_acc))
    print("Confusion Matrix:")
    print(str(best_conf))
    sys.stdout = sys.__stdout__

    if args.test_data is not None:
        for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features"]:
            features, _ = build_features(eval_data, feat_set, args.binning, feat_set+'-'+'test')
            if feat_set is best_feat_set:
                sys.stdout = open(args.write_fname, 'w')
                for pred in best_classifier.classify_many([feat[0] for feat in features]):
                    print(str(pred) + '\n')
                sys.stdout = sys.__stdout__







if __name__ == "__main__":
    main()




