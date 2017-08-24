import os
#import glob
#import re
#import stanford_corenlp_pywrapper
#import utils
import logging
#import sys
import numpy
#import collections
import random
#from multiprocessing import Process, Manager
#from gensim.models import Phrases
#from gensim.corpora import Dictionary
#from gensim.models import CoherenceModel, LdaModel, HdpModel, LdaMulticore
#from gensim.models.wrappers import LdaMallet
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')


def fix_item_path(item, section):
    return os.path.join(section, item[item.rfind('/')+1:])


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data)
        corpus, dictionary, texts, time_slices = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

        model_filepath = config.TRAIN_PARAMETERS[section][1]
        print(model_filepath)
        assert os.path.exists(model_filepath)
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=False)
        print(c_v, u_mass)
        for item, text in data:
            item = fix_item_path(item, section)
            # TODO
