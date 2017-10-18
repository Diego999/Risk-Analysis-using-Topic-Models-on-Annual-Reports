import pickle
import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from operator import itemgetter
import pandas as pd
import warnings
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
import seaborn as sns
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
    colors = np.array(["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in sns.color_palette(n_colors=nb_keys)])
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # logging.getLogger().setLevel(logging.INFO)
    np.random.seed(config.SEED) # To choose the same set of color
    random.seed(config.SEED)

    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]
    for section in sections_to_analyze:
        input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/') + 1:] + config.SUFFIX_DF + '.pkl')
        data = pd.read_pickle(input_file)

        # Visualize sector & industry
        for key in ['sector', 'industry']:
            data_filtered = data[pd.isnull(data[key]) == False]
            for provider in set([x['provider_sector_industry'] for _, x in data_filtered.iterrows()]):
                data_filtered = data[data['provider_sector_industry'] == provider]
                # Aggregate all companies' reports to a single file
                ids = [(x.split(config.CIK_COMPANY_NAME_SEPARATOR)) for x, _ in data_filtered.iterrows()]
                assert len([1 for x in ids if len(x) > 2]) == 0
                ids_dict = {}
                for comp, rep in ids:
                    if comp not in ids_dict:
                        ids_dict[comp] = []
                    ids_dict[comp].append(rep)

                data_filtered_unique_comp = data_filtered[0:0]
                for comp, reps in ids_dict.items():
                    data_filtered_unique_comp_local = data_filtered[0:0]
                    for rep in reps:
                        index = comp + config.CIK_COMPANY_NAME_SEPARATOR + rep
                        data_filtered_unique_comp_local = data_filtered_unique_comp_local.append(data_filtered.ix[[index]])
                    assert len(data_filtered_unique_comp_local) == len([x[key] for _, x in data_filtered_unique_comp_local.iterrows()])

                    avg_topic = [(topic_id, topic) for topic_id, topic in enumerate(np.mean(get_topics(data_filtered_unique_comp_local), axis=0))]
                    data_filtered_unique_comp = data_filtered_unique_comp.append(data_filtered.ix[[index]])
                    data_filtered_unique_comp.iloc[-1]['topics'] = pd.Series(avg_topic)
                data_filtered = data_filtered_unique_comp

                docs, vals = get_key_vals(data_filtered, key)
                topics = get_topics(data_filtered)

                proj_filepath = section + '_' + 'unique_tsne' + '_' + key + '_' + provider
                model, proj_topics = train_or_load_tsne(proj_filepath, topics, provider, seed=config.SEED)

                five_highest_topics = visualize.get_five_highest_topics(topics)
                colors, color_keys, key_values = get_colors(data_filtered, key)
                company_names = visualize.get_company_names(docs)

                # Global plot
                plot(proj_topics, docs, company_names, five_highest_topics, key_values, len(vals), section + ' (all {} with {})'.format(key, provider), colors, color_keys, proj_filepath + '_all_{}_{}'.format(key, provider), len(colors), key)

                # Global plot for all companies per key
                for t in get_vals_per_key(proj_topics, data_filtered, vals, five_highest_topics, colors, color_keys, key_values, company_names, key):
                    key_val, reduced_proj_lda, reduced_docs, reduced_vals, reduced_five_highest_topics, reduced_colors, reduced_color_keys, reduced_key_values, reduced_company_names = t
                    plot(reduced_proj_lda, reduced_docs, reduced_company_names, reduced_five_highest_topics, reduced_key_values, len(reduced_color_keys), section + ' ({} {} {})'.format(key, key_val, provider), reduced_colors, reduced_color_keys, proj_filepath + '_{}_{}'.format(key_val.replace('/','_'), provider), len(colors), key)
