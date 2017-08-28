import os
import logging
import numpy
import random
import pickle
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')


def fix_item_path(item, section):
    return os.path.join(section, item[item.rfind('/')+1:])


def get_topics(num_topics, model, idx_to_lemma, output):
    # Get definition of the topics
    topics = []
    for topic_id in range(0, num_topics):
        lemmas_probs = model.get_topic_terms(topicid=topic_id, topn=None)  # to gather all terms
        lemmas_probs = [(idx_to_lemma[word_id], prob) for word_id, prob in lemmas_probs]
        topics.append(lemmas_probs)

    # Save topics
    with open(output, 'wb') as fp:
        pickle.dump(topics, fp)
    with open(output + '.txt', 'w', encoding='utf-8') as fp:
        for topic_id, lemmas_probs in enumerate(topics):
            fp.write('Topic {}\n'.format(topic_id))
            for l, p in lemmas_probs:
                fp.write('\t{}\t{}\n'.format(l, p))

    return topics


def get_terms_topics(model, idx_to_lemma, output):
    # Get terms distribution w.r.t. to each topics
    terms_topics = []
    for lemma_id, lemma in idx_to_lemma.items():
        terms_topics.append((lemma, model.get_term_topics(lemma_id, 1e-10)))

    # Save terms
    with open(output, 'wb') as fp:
        pickle.dump(terms_topics, fp)
    with open(output + '.txt', 'w', encoding='utf-8') as fp:
        for lemma, topic_distr in terms_topics:
            fp.write('{}\n'.format(lemma))
            for topic_id, topic_prob in topic_distr:
                fp.write('\t{}\t{}\n'.format(topic_id, topic_prob))

    return terms_topics


def get_document_topic_distribution(num_topics, data, corpus, model, output):
    documents_topic_distr = []
    for i, (item, text) in enumerate(data):
        # item = fix_item_path(item, section) # To filter root path

        # Get % of each topic per document
        topic_distr = model.get_document_topics(corpus[i], 1e-10)
        assert len(topic_distr) == num_topics
        documents_topic_distr.append((item, topic_distr))

    with open(output, 'wb') as fp:
        pickle.dump(documents_topic_distr, fp)
    with open(output + '.txt', 'w', encoding='utf-8') as fp:
        for item, distr in documents_topic_distr:
            fp.write(item + '\t' + ' '.join([str(x) for x in distr]) + '\n')

    return documents_topic_distr


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    if not os.path.exists(config.DATA_DISTRIBUTION_FOLDER):
        os.makedirs(config.DATA_DISTRIBUTION_FOLDER)

    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data)
        corpus, dictionary, texts, time_slices, data = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

        model_filepath = config.TRAIN_PARAMETERS[section][1]
        assert os.path.exists(model_filepath)
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

        num_topics = config.TRAIN_PARAMETERS[section][0]
        topics = get_topics(num_topics, model, dictionary.id2token, config.TRAIN_PARAMETERS[section][5])
        terms_topics = get_terms_topics(model, dictionary.id2token, config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.TERMS_EXTENSION))
        documents_topic_distr = get_document_topic_distribution(num_topics, data, corpus, model, config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.DOCS_EXTENSION))
