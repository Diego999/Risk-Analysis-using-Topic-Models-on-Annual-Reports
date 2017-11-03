import numpy as np
import random
import os
import pandas as pd
import warnings
import utils
import scipy.cluster.hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from scipy.cluster import hierarchy
import scipy
from scipy.stats import entropy
config = __import__('0_config')
visualize = __import__('4e_visualize')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # logging.getLogger().setLevel(logging.INFO)
    np.random.seed(config.SEED) # To choose the same set of color
    random.seed(config.SEED)

    companies = set()
    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]
    for section in sections_to_analyze:
        input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/') + 1:] + config.SUFFIX_DF + '.pkl')
        data_filtered = pd.read_pickle(input_file)

        top_n_topic_freq_per_year = {}
        n = 10
        # Aggregate reports by time
        year_topics = [(utils.year_annual_report_comparator(int(file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1])), x['topics']) for file, x in data_filtered.iterrows()]
        year_counts = {}
        for y, t in year_topics:
            if y not in year_counts:
                year_counts[y] = 0
            year_counts[y] += 1

        for y, t in year_topics:
            top_topics = sorted(t, key=lambda x:x[1], reverse=True)[:n]
            if y not in top_n_topic_freq_per_year:
                top_n_topic_freq_per_year[y] = {}
            for topic_id, prob in top_topics:
                if topic_id not in top_n_topic_freq_per_year[y]:
                    top_n_topic_freq_per_year[y][topic_id] = 0
                top_n_topic_freq_per_year[y][topic_id] += 1.0/year_counts[y]

        print('Top {} topic(s)'.format(n))
        for year, topics in sorted(top_n_topic_freq_per_year.items(), key=lambda x:x[0]):
            print('\t{}: '.format(year) + ' '.join([str(t+1) + '(' + str(v) + ')' for t, v in sorted(topics.items(), key=lambda x:x[1], reverse=True)][:n]))

        unique_topics_set = set()
        for i, (year, topics) in enumerate(sorted(top_n_topic_freq_per_year.items(), key=lambda x:x[0])):
            topics = [(t, v) for t, v in sorted(topics.items(), key=lambda x:x[1], reverse=True)][:n]
            topic_ids = [t for t,v in topics]
            for t in [t for t,v in topics]:
                unique_topics_set.add(t)

        ind = 15*np.arange(len(unique_topics_set))  # the x locations for the groups
        unique_topics = sorted(list(unique_topics_set))
        width = 0.5

        fig, ax = plt.subplots()
        rects = []
        for i, (year, topics) in enumerate(sorted(top_n_topic_freq_per_year.items(), key=lambda x: x[0])):
            topics = [(t,  v) for t, v in sorted(topics.items(), key=lambda x:x[1], reverse=True) if t in unique_topics_set]
            topics_set = set([t for t, v in topics])
            for t in unique_topics:
                if t not in topics_set:
                    topics.append([t, 0])
            topics = [v for k,v in sorted(topics, key=lambda x:x[0])]
            rect = ax.bar(ind + n*width + i*width, topics, width)
            rects.append(rect)
        ax.set_xticklabels(tuple(['Topic #{}'.format(t+1) for t in unique_topics]), rotation=-90)
        ax.set_xticks(ind + n*width + len(top_n_topic_freq_per_year.items())*width / 2)
        ax.set_ylabel('Ratio of reports having these topics')
        ax.set_title('Occurences for most trendy topics for all years')
        ax.legend(tuple([rect[0] for rect in rects]), tuple([y for y in sorted(list(top_n_topic_freq_per_year.keys()))]))

        plt.show()
