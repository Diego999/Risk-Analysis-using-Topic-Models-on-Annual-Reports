import pickle
import numpy as np
import random
import os
import matplotlib
#matplotlib.use('Agg')
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from operator import itemgetter
from scipy.spatial.distance import cosine
import pandas as pd
import warnings
import bokeh.plotting as bp
import utils
from bokeh.plotting import save
from bokeh.models import HoverTool
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from scipy.cluster import hierarchy
import scipy
from collections import defaultdict
config = __import__('0_config')
visualize = __import__('4e_visualize')


def get_topics(data):
    id_topics = [x['topics'] for _, x in data.iterrows()]
    return np.array([[xxx[1] for xxx in xx] for xx in id_topics])


def get_key_vals(data, key):
    temp = [(file, x[key]) for file, x in data.iterrows()]
    return [x[0] for x in temp], [x[1] for x in temp]


# From Scikit: t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples
# This allows t-SNE to be particularly sensitive to local structure and has a few other advantages over existing techniques:
# - Revealing the structure at many scales on a single map
# - Revealing data that lie in multiple, different, manifolds or clusters
# - Reducing the tendency to crowd points together at the center
def train_or_load_tsne(tsne_filepath, topics, provider, seed=config.SEED):
    tsne_topics = None
    tsne_model = None
    if os.path.exists(tsne_filepath):
        with open(tsne_filepath, 'rb') as fp:
            tsne_topics = pickle.load(fp)
        with open(tsne_filepath + '_tsne_model_{}'.format(provider), 'rb') as fp:
            tsne_model = pickle.load(fp)
    else:
        tsne_model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=50.0, n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-7, verbose=1, random_state=seed, angle=.4, init='pca')
        tsne_topics = tsne_model.fit_transform(topics)
        with open(tsne_filepath + '_tsne_model_{}'.format(provider), 'wb') as fp:
            pickle.dump(tsne_model, fp)
        with open(tsne_filepath, 'wb') as fp:
            pickle.dump(tsne_topics, fp)

    return tsne_model, tsne_topics


def train_or_load_pca(pca_filepath, topics, provider, seed=config.SEED):
    pca_lda = None
    pca_model = None
    if os.path.exists(pca_filepath):
        with open(pca_filepath, 'rb') as fp:
            pca_lda = pickle.load(fp)
        with open(pca_filepath + '_pca_model_{}'.format(provider), 'rb') as fp:
            pca_model = pickle.load(fp)
    else:
        pca_model = PCA(n_components=2, svd_solver='auto', random_state=seed)
        pca_lda = pca_model.fit_transform(topics)
        with open(pca_filepath + '_pca_model_{}'.format(provider), 'wb') as fp:
            pickle.dump(pca_model, fp)
        with open(pca_filepath, 'wb') as fp:
            pickle.dump(pca_lda, fp)

    return pca_model, pca_lda


def get_colors(data, key):
    keys = sorted(list({x[key] for _, x in data.iterrows()}))
    k2i = {k:i for i, k in enumerate(keys)}
    nb_keys = len(keys)
    x = np.random.random(size=nb_keys) * 100
    y = np.random.random(size=nb_keys) * 100
    z = np.random.random(size=nb_keys) * 100
    colors = np.array(["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b in zip(2.55 * x, 2.55 * y, 2.55 * z)])
    color_keys = [k2i[x[key]] for _, x in data.iterrows()]
    color_vals = [x[key] for _, x in data.iterrows()]

    return colors, color_keys, color_vals


def plot(proj_lda, docs, company_names, five_highest_topics, key_values, nb_samples, title, colors, color_keys, filename, nb_topics, key):
    # Plot
    plot_lda = bp.figure(plot_width=1820, plot_height=950,
                         title=title + ' ({} sample{}, {} {})'.format(nb_samples, 's' if nb_samples > 1 else '', nb_topics, key),
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    plot_lda.scatter(x=proj_lda[:nb_samples, 0], y=proj_lda[:nb_samples, 1],
                     color=colors[color_keys][:nb_samples],
                     source=bp.ColumnDataSource({
                         "X":proj_lda[:nb_samples, 0],
                         "Y":proj_lda[:nb_samples, 1],
                         "5_highest_topics": five_highest_topics[:nb_samples] if nb_samples > 1 else [five_highest_topics[:nb_samples]],
                         key: key_values[:nb_samples] if nb_samples > 1 else [key_values[:nb_samples]],
                         "file": docs[:nb_samples] if nb_samples > 1 else [docs[:nb_samples]],
                         "company": company_names[:nb_samples] if nb_samples > 1 else [company_names[:nb_samples]]
                     })
                     )

    # Hover tool
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"X":"@X", "Y":"@Y", key: "@"+key, "5 highest topics": "@5_highest_topics", "Filename": "@file", "Company": "@company"}

    save(plot_lda, '{}.html'.format(filename))


def get_vals_per_key(proj_topics, data, vals, five_highest_topics, colors, color_keys, key_values, company_names, key):
    values_indices = {}
    for i, (_, x) in enumerate(data.iterrows()):
        value = x[key]
        if value not in values_indices:
            values_indices[value] = []
        values_indices[value].append(i)

    assert sum([len(x) for x in values_indices.values()]) == len(docs)

    for value, indices in values_indices.items():
        yield value, \
              np.take(proj_topics, indices, axis=0), \
              itemgetter(*indices)(docs), \
              itemgetter(*indices)(vals), \
              [five_highest_topics[i] for i in indices], \
              colors, \
              (list(itemgetter(*indices)(color_keys)) if len(indices) > 1 else [itemgetter(*indices)(color_keys)]), \
              [key_values[i] for i in indices], \
              [company_names[i] for i in indices]


def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes


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