import os
import numpy
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')

if __name__ == "__main__":
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    if not os.path.exists(config.DATA_DISTRIBUTION_FOLDER):
        os.makedirs(config.DATA_DISTRIBUTION_FOLDER)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data)
        corpus, dictionary, texts, time_slices, data = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

        model_filepath = config.TRAIN_PARAMETERS[section][1]
        assert os.path.exists(model_filepath)
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

        num_topics = config.TRAIN_PARAMETERS[section][0]
        cols = 11
        rows = int(num_topics/cols)
        '''for t in range(num_topics):
            plt.figure()
            x = model.show_topic(t, 50)
            x = {k:v for k,v in x}  # convert from string to float
            plt.imshow(WordCloud(background_color='white').fit_words(x))
            plt.axis("off")
            plt.title("Topic #" + str(t+1))
            plt.savefig("1a_risk_factors_Topic_" + str(t+1) + '.png')'''

        plt.figure(figsize=(rows, cols))
        gs1 = gridspec.GridSpec(rows, cols)
        gs1.update(wspace=.0, hspace=.25)
        for t in range(num_topics):
            ax1 = plt.subplot(gs1[t])
            plt.axis('off')
            plt.title("Topic #" + str(t+1))
            ax1.set_aspect('equal')
            x = model.show_topic(t, 50)
            x = {k: v for k, v in x}  # convert from string to float
            plt.imshow(WordCloud(background_color='white').fit_words(x), interpolation='nearest', aspect='auto')
        plt.show()