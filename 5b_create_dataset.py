import utils
import os
import glob
import numpy as np
import math
analyze_topics_static = __import__('4a_analyze_topics_static')
infer_topics = __import__('4d_infer_topics')
config = __import__('0_config')


def get_topic_per_documents(section, data_):
    data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data_)
    corpus, dictionary, texts, time_slices, data = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

    model_filepath = config.TRAIN_PARAMETERS[section][1]
    assert os.path.exists(model_filepath)
    model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

    num_topics = config.TRAIN_PARAMETERS[section][0]
    return infer_topics.get_document_topic_distribution(num_topics, data, corpus, model, config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.DOCS_EXTENSION))
stocks = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_STOCKS_FOLDER))}
ni_seqs = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_NI_SEQ_FOLDER))}
sec_inds = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_SEC_IND_FOLDER))}
if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        topics = {x[0][x[0].rfind('/')+1:-4]:x[1] for x in get_topic_per_documents(section, data)}

        data = {x[0].split('/')[-1].split('.')[0]:{'text': x[1]} for x in data}
        # add topics
        for file, attributes in data.items():
            data[file]['topics'] = topics[file]