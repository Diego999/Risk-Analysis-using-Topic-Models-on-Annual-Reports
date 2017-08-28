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
        assert os.path.exists(model_filepath)
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

        get_topics(config.TRAIN_PARAMETERS[section][0], model, dictionary.id2token, config.TRAIN_PARAMETERS[section][5])

        for item, text in data:
            item = fix_item_path(item, section)

            # Get % of each topic per document
            # TODO model.get_document_topics
