
import nltk
import re
import word_category_counter
import data_helper
import os, sys
from word2vec_extractor import Word2vecExtractor
DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)

w2vecmodel = "data/glove-w2v.txt"
w2v = None

def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token if token not in nltk.corpus.stopwords.words('english') else None

    else:

        ###     YOUR CODE GOES HERE
        normalized_token = token.lower() if token not in nltk.corpus.stopwords.words('english') else None

    return normalized_token

def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []


    ###     YOUR CODE GOES HERE

    if should_normalize:
        text = text.lower()

    tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(text)]
    tagged_sentences = [tup for sent in tagged_sentences for tup in sent]
    words = [tup[0] for tup in tagged_sentences]
    tags = [tup[1] for tup in tagged_sentences]

    return words, tags

def get_pos_features(tags, binning):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :param binning: whether if we want to bin the values or not
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE
    uni_fd = nltk.FreqDist(tags)
    bi_fd = nltk.FreqDist(nltk.bigrams(tags))
    feature_vectors = {'UNI_' + k: (bin(v) if binning else v) for (k, v) in uni_fd.items()}

    for (k, v) in bi_fd.items():
        feature_vectors['BI_' + k[0] + '_' + k[1]] = (bin(v) if binning else v)

    return feature_vectors

def get_ngram_features(tokens, binning):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :param binning: whether if we want to bin the values or not
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE
    uni_fd = nltk.FreqDist(tokens)
    bi_fd = nltk.FreqDist(nltk.bigrams(tokens))
    feature_vectors = {'UNI_' + k: (bin(v) if binning else v) for (k, v) in uni_fd.items()}
    for (k, v) in bi_fd.items():
        feature_vectors['BI_' + k[0] + '_' + k[1]] = (bin(v) if binning else v)

    return feature_vectors

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

def get_liwc_features(words, binning):
    """
    Adds a simple LIWC derived feature

    :param words:
    :param binning: whether if we want to bin the values or not
    :return:
    """

    # TODO: binning

    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)

    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    for (key, value) in liwc_scores.items():
        feature_vectors[key] = (bin(value) if binning else value)

    #if positive_score > negative_score:
    #    feature_vectors["liwc:positive"] = 1
    #else:
    #    feature_vectors["liwc:negative"] = 1

    return feature_vectors

def get_word_embedding_features(text):
    global w2v
    if w2v is None:
        print("loading word vectors ...", w2vecmodel)
        w2v = Word2vecExtractor(w2vecmodel)
    feature_dict = w2v.get_doc2vec_feature_dict(text)
    return feature_dict



FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "only_liwc",
                "word_embedding"}


def get_features_category_tuples(category_text_dict, feature_set, binning=False):
    """

    You will might want to update the code here for the competition part.

    This method is going to return a feature vector depending on the feature set indicated in feature set.
    Also, during the process the stopwords are gonna be eliminated from the feature vector.
    :param category_text_dict:
    :param feature_set:
    :param binning: whether if we want to bin the values or not
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)
    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            delete_list = []
            feature_vectors = {}

            ###     YOUR CODE GOES HERE
            for i in range(len(words)):
                nor_word = normalize(words[i])
                if nor_word is not None:
                    words[i] = nor_word
                else:
                    delete_list.append(i)

            for i in reversed(delete_list):
                del words[i]
                del tags[i]
            feature_vectors = get_ngram_features(words, binning)
            if feature_set is "word_pos_features":
                feature_vectors.update(get_pos_features(tags, binning))
            if feature_set is "word_pos_liwc_features":
                feature_vectors.update(get_pos_features(tags, binning))
                feature_vectors.update(get_liwc_features(words, binning))
            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts






if __name__ == "__main__":
    print("hello world!")

