import pickle
import numpy as np
import random
import os
from sklearn.manifold import TSNE
from itertools import combinations
import datetime
import utils
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
config = __import__('0_config')


def extract_section(section):
    return section[section.rfind('/')+1:section.rfind('_')]


def extract_item(item):
    return item[item.rfind('/') + 1:]


def get_embeddings(paths_num_topics):
    embeddings = {}
    for p, num_topics in paths_num_topics:
        section = extract_section(p)

        documents_topic_distr = []
        with open(p, 'rb') as fp:
            documents_topic_distr = pickle.load(fp)

        embeddings[section + '_{}'.format(num_topics)] = {extract_item(item):[val for topic_id, val in emb] for item, emb in documents_topic_distr}

    return embeddings


def combine_embeddings(embeddings):
    for r in range(2, len(embeddings.keys())+1):
        for combs in combinations(list(embeddings.keys()), r):
            embedding_keys = list(combs) # Assure to keep the same order
            keys = [set(embeddings[key].keys()) for key in embedding_keys]

            final_intersection = keys[0]
            for key in keys[1:]:
                final_intersection = final_intersection & key

            final_key = '|'.join(embedding_keys)
            embeddings[final_key] = {}
            for key1 in embedding_keys:
                for key2 in final_intersection:
                    if key2 not in embeddings[final_key]:
                        embeddings[final_key][key2] = embeddings[key1][key2][:] # Deep copy
                    else:
                        embeddings[final_key][key2] += embeddings[key1][key2] # Merge vector together

    return embeddings


def convert_to_matrices(embeddings):
    # Transform to have a matrix by key where each row is a fix document and cols are embeddings
    embeddings_matrices = {}
    for k, v in embeddings.items():
        documents = sorted(list(v.keys()))
        vals = np.array([embeddings[k][d] for d in documents])
        embeddings_matrices[k] = (documents, vals)

    return embeddings_matrices


def train_or_load_tsne(tsne_filepath, seed=config.SEED):
    tsne_lda = None
    if os.path.exists(tsne_filepath):
        with open(tsne_filepath, 'rb') as fp:
            tsne_lda = pickle.load(fp)
    else:
        tsne_model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=50.0, n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-7, verbose=1, random_state=seed, angle=.4, init='pca')
        tsne_lda = tsne_model.fit_transform(vals)
        with open(tsne_filepath, 'wb') as fp:
            pickle.dump(tsne_lda, fp)

    return tsne_lda


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.INFO)
    np.random.seed(config.SEED) # To choose the same set of color
    random.seed(config.SEED)

    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]
    paths_num_topics = [(config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.DOCS_EXTENSION), config.TRAIN_PARAMETERS[section][0]) for section in sections_to_analyze]
    if not os.path.exists(config.DATA_TSNE_FOLDER):
        os.makedirs(config.DATA_TSNE_FOLDER)

    embeddings = get_embeddings(paths_num_topics)
    embeddings = combine_embeddings(embeddings)
    embeddings_matrices = convert_to_matrices(embeddings)

    for section in sorted(list(embeddings_matrices.keys()), key=lambda x:len(x)):
        docs, vals = embeddings_matrices[section]
        nb_samples = len(docs)

        filename = os.path.join(config.DATA_TSNE_FOLDER, section)
        # Pickle cannot dump/load such filepath
        if len(filename) > 150:
            filename = filename[:150]

        tsne_lda = train_or_load_tsne(filename + '.pkl')

        # Generate info for the hover tool
        five_highest_topics = []
        for embs in vals:
            # https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
            ind = np.argpartition(embs, -5)[-5:]
            sorted_ind = [str(x) for x in ind[np.argsort(embs[ind])]]
            five_highest_topics.append(' '.join(sorted_ind))

        nb_years = datetime.datetime.now().year - config.START_YEAR + 1
        x = np.random.random(size=nb_years) * 100
        y = np.random.random(size=nb_years) * 100
        z = np.random.random(size=nb_years) * 100
        colors = np.array(["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b in zip(2.55 * x,  2.55 * y, 2.55 * z)])
        color_keys = [utils.year_annual_report_comparator(int(d.split('-')[-2])) - config.START_YEAR for d in docs] # 0 = config.START_YEAR

        company_names = [os.path.join(config.DATA_AR_FOLDER, os.path.join(d.split('_')[0]), config.NAME_FILE_PER_CIK) for d in docs]
        for i, n in enumerate(company_names):
            with open(n, 'r', encoding='utf-8') as fp:
                company_names[i] = ' | '.join([l.strip() for l in fp if len(l.strip()) > 0])

        # Plot
        plot_lda = bp.figure(plot_width=1820, plot_height=950,
                             title=section + '({} samples)'.format(nb_samples),
                             tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                             x_axis_type=None, y_axis_type=None, min_border=1)

        year_values = np.full((nb_samples), config.START_YEAR) + color_keys[:nb_samples]
        plot_lda.scatter(x=tsne_lda[:nb_samples, 0], y=tsne_lda[:nb_samples, 1],
                         color=colors[color_keys][:nb_samples],
                         source=bp.ColumnDataSource({
                             "5_highest_topics": five_highest_topics[:nb_samples],
                             "year": year_values[:nb_samples],
                             "file":docs[:nb_samples],
                             "company":company_names[:nb_samples]
                         })
                         )

        # Hover tool
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"Year": "@year", "5 highest topics": "@5_highest_topics", "Filename":"@file", "Company":"@company"}

        save(plot_lda, '{}.html'.format(filename))
