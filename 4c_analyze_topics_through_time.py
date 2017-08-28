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
from gensim.models import ldaseqmodel
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')


try:
    import pyLDAvis
    CAN_VISUALIZE = True
except ImportError:
    CAN_VISUALIZE = False


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
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
        corpus, dictionary, texts, time_slices, _ = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

        # Load model
        model_file = config.TRAIN_PARAMETERS[section][1]
        assert os.path.exists(model_file)
        num_topics = config.TRAIN_PARAMETERS[section][0]
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, num_topics=num_topics, alpha='symmetric', eta='auto', chunksize=2000, decay=0.5, offset=1.0, passes=10, iterations=400, eval_every=10, model_file=model_file, only_viz=config.DO_NOT_COMPUTE_COHERENCE)
        print('c_v:' + str(round(c_v, 4)) + ', cu:' + str(round(u_mass, 4)))

        dyn_model = ldaseqmodel.LdaSeqModel(initialize='ldamodel', lda_model=model, time_slice=time_slices, corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=config.SEED)
        filename = config.TRAIN_PARAMETERS[section][3]
        dyn_model.save(filename)

        for t in range(0, len(time_slices)):
            doc_topic, topic_term, doc_lengths, term_frequency, vocab = dyn_model.dtm_vis(time=t, corpus=corpus)
            prepared = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)

            loc_dot_ext = filename.rfind('.')
            year = config.START_YEAR + t
            filename = filename[:loc_dot_ext] + '_{}'.format(year) + filename[filename.rfind('.'):]
            pyLDAvis.save_html(prepared, filename + '.html')