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
import scipy.spatial as sp, scipy.cluster.hierarchy
from bokeh.plotting import save
from bokeh.models import HoverTool
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from scipy.cluster import hierarchy
import scipy
from scipy.stats import entropy
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