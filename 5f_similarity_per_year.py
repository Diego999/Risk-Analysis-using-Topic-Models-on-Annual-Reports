import numpy as np
import random
import os
from scipy.spatial.distance import cosine
import pandas as pd
import warnings
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from scipy.cluster import hierarchy
config = __import__('0_config')
visualize = __import__('4e_visualize')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # logging.getLogger().setLevel(logging.INFO)
    np.random.seed(config.SEED) # To choose the same set of color
    random.seed(config.SEED)

    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]
    for section in sections_to_analyze:
        input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/') + 1:] + config.SUFFIX_DF + '.pkl')
        data = pd.read_pickle(input_file)

        docs, _ = get_key_vals(data, 'topics')
        topics = get_topics(data)

        centroids = {}
        for d, t in zip(docs, topics):
            year = utils.year_annual_report_comparator(int(d.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1]))
            if year not in centroids:
                centroids[year] = []
            centroids[year].append(t)

        keys_centroids = sorted(list(centroids.keys()))
        for i in range(len(keys_centroids)):
            print('in ' + str(keys_centroids[i]) + ' there are ' + str(len(centroids[keys_centroids[i]])) + ' reports')
            centroids[keys_centroids[i]] = np.mean(np.array(centroids[keys_centroids[i]]), axis=0)
        print('')

        # Cosine similarity among centroids
        distances_centroids = {}
        for i in range(len(keys_centroids)):
            local_dist = []
            for j in range(len(keys_centroids)):
                local_dist += [1.0 - cosine(centroids[keys_centroids[i]], centroids[keys_centroids[j]])]
            distances_centroids[keys_centroids[i]] = copy.deepcopy(local_dist)

        final_out = ''
        for i in keys_centroids:
            out = ''
            for j in range(len(keys_centroids)):
                out += '{:.2f} '.format(distances_centroids[i][j])
            final_out += out + '\n'
        print(final_out)
        distances_centroids = pd.DataFrame(distances_centroids, columns=keys_centroids, index=keys_centroids)
        print(distances_centroids)
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=15, center="dark")

        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(distances_centroids, cmap=cmap, vmax=1.0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        ax.get_figure().savefig('cosine_similarity_centroids.png')
        for method in ['average']:
            clustergrid = sns.clustermap(distances_centroids, cmap=cmap, linewidths=.75, figsize=(13, 13), method=method)
            plt.savefig('hierarchical_clustering_cos_sim_{}.png'.format(method))
            plt.figure()
            dn = hierarchy.dendrogram(clustergrid.dendrogram_col.linkage, labels=np.array(keys_centroids), orientation='right')
            plt.savefig('hierarchical_1D_clustering_cos_sim_{}.png'.format(method))

        # Cluster for pearson-correlation
        raw_vectors = np.array([centroids[keys_centroids[i]] for i in range(len(keys_centroids))])
        coeffs = np.corrcoef([*raw_vectors])
        final_out = ''
        for c in coeffs.tolist():
            out = ''
            for cc in c:
                out += '{:2.2f} '.format(cc)
            final_out += out + '\n'
        print(final_out)

        coeffs = pd.DataFrame(coeffs, columns=keys_centroids, index=keys_centroids)
        for method in ['average']:
            clustergrid = sns.clustermap(distances_centroids, cmap=cmap, linewidths=.75, figsize=(13, 13), method=method)
            plt.savefig('hierarchical_clustering_pearson_{}.png'.format(method))
            plt.figure()
            dn = hierarchy.dendrogram(clustergrid.dendrogram_col.linkage, labels=np.array(keys_centroids), orientation='right')
            plt.savefig('hierarchical_1D_clustering_pearson_{}.png'.format(method))
        plt.show()