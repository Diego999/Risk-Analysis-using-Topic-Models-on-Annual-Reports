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

def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5


# Use KL divergence
def d_jsd(v1, v2, mean_vector):
    return np.sqrt(0.5*scipy.stats.entropy(v1, mean_vector) + 0.5*scipy.stats.entropy(v2, mean_vector))


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

        # Aggregate reports by time
        year_topics = [(utils.year_annual_report_comparator(int(file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1])), np.array([xx[1] for xx in x['topics']])) for file, x in data_filtered.iterrows()]
        topics_year = {}
        for y, t in year_topics:
            if y not in topics_year:
                topics_year[y] = []
            topics_year[y].append(t)
        for y, t in topics_year.items():
            topics_year[y] = np.mean(np.array(t), axis=0)

        plt.figure()
        topics_year_df = pd.DataFrame(topics_year, columns=sorted(topics_year.keys()), index=['Topic #' + str(i + 1) for i in range(len(year_topics[0][1]))]).T
        sns.heatmap(topics_year_df, cmap=sns.color_palette("RdBu_r", 14), center=0.0, linewidths=.75)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)
        plt.title('Topics by year')

        # hierarchical clustering per year
        distances_centroids = {}
        topics_year_mean_all = np.mean(np.array([x[1] for x in topics_year.items()]), axis=0)
        elems = sorted(topics_year.items(), key=lambda x:x[0])
        for i, (y1, t1) in enumerate(elems):
            local_dist = []
            for y2, t2 in elems:
                local_dist += [d_jsd(t1, t2, topics_year_mean_all)]
            distances_centroids[y1] = copy.deepcopy(local_dist)

        final_out = ''
        for i, _ in elems:
            out = ''
            for j in range(len(distances_centroids[i])):
                out += '{:.2f} '.format(distances_centroids[i][j])
            final_out += out + '\n'
        print(final_out)

        plt.figure()
        distance_matrix = []
        for k, v in elems:
            distance_matrix.append(distances_centroids[k])
        distance_matrix = np.array(distance_matrix)
        linkage_distance_matrix = scipy.cluster.hierarchy.linkage(distance_matrix, method='average')
        distances_centroids_df = pd.DataFrame(distance_matrix, index=[x[0] for x in elems], columns=[x[0] for x in elems])
        clustergrid = sns.heatmap(distances_centroids_df, cmap=sns.color_palette("RdBu_r", 14), center=0.0, linewidths=.75)
        plt.title('Squared Jensen-Shannon divergence distance between years over all reports')
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)

        plt.figure()
        dn = hierarchy.dendrogram(linkage_distance_matrix, labels=np.array([x[0] for x in elems]), orientation='right')
        plt.title('Hierarchical dendogram for year over all reports')

        # Stack plot with time vs topic proportion
        fig, ax = plt.subplots()
        x = [x for x, _ in elems]
        y = [xx for _, xx in elems]
        y_t = np.transpose(np.array(y)) # topic vs time
        ax.stackplot(x, *y_t, labels=['Topic #' + str(i + 1) for i in range(len(year_topics[0][1]))])
        plt.xlabel('Year')
        plt.xticks(range(elems[0][0], elems[-1][0]+1), rotation=-90)
        plt.ylabel('Distribution')
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.title('Evolution of topics through time over all reports')
        labels = ['Topic #' + str(i + 1) for i in range(len(year_topics[0][1]))]
        offset = 0
        for i in range(0, len(labels)):
            if i % 2 == 0:
                ax.text(x[-1]+0.5, offset + y_t[i][-1]/2, labels[i])
            offset += y_t[i][-1]

        plt.show()