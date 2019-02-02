
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




def get_word_embedding_features(text):
    global w2v
    if w2v is None:
        print("loading word vectors ...", w2vecmodel)
        w2v = Word2vecExtractor(w2vecmodel)
    feature_dict = w2v.get_doc2vec_feature_dict(text)
    return feature_dict



FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "only_liwc",
                "word_embedding"}









if __name__ == "__main__":
    print("hello world!")

