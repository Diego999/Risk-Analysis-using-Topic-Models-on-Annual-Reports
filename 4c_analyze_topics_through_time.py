import os
import glob
import re
import stanford_corenlp_pywrapper
import utils
import logging
import sys
import numpy
import collections
import random
from multiprocessing import Process, Manager
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, HdpModel, LdaMulticore
from gensim.models.wrappers import LdaMallet
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')




if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)
    if not os.path.exists(config.MODEL_FOLDER):
        os.makedirs(config.MODEL_FOLDER)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data)
        corpus, dictionary, texts = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

        # Load a model
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=config.ITEM_1A_MODEL, only_viz=config.DO_NOT_COMPUTE_COHERENCE)
        print('c_v:' + str(round(c_v, 4)) + ', cu:' + str(round(u_mass, 4)))
        analyze_topics_static.visualize(model, corpus, dictionary)