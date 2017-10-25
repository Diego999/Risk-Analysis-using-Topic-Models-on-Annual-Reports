import utils
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
import seaborn as sns
config = __import__('0_config')
from sklearn.manifold import TSNE


def load_embeddings(file_name):
    wv = []
    vocabulary = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        header = True
        for line in fp:
            if header:
                header = False
            else:
                line = line.strip().split(' ')
                word, emb = line[0], line[1:]
                emb = np.array([float(x) for x in emb])
                wv.append(emb)
                vocabulary.append(word)

    return np.array(wv), vocabulary


def get_colors(ws, colors=np.array([sns.xkcd_rgb["purple"], sns.xkcd_rgb["brown"], sns.xkcd_rgb["red"], sns.xkcd_rgb["grey"], sns.xkcd_rgb["green"], sns.xkcd_rgb["pink"], sns.xkcd_rgb["orange"]])):
    keys = sorted(list(set(ws)))
    k2i = {k:i for i, k in enumerate(keys)}
    nb_keys = len(keys)
    color_keys = [k2i[sent] for sent in ws]
    color_vals = ws

    return colors, color_keys, color_vals


def plot(proj_tsne, vocabulary, wv, ws, nb_samples, title, colors, color_keys, filename):
    # Plot
    plot_we = bp.figure(plot_width=1820, plot_height=950,
                         title=title + ' ({} words of dim {}, {} sentiment categories)'.format(nb_samples, len(wv[0]), len(set(ws))),
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    plot_we.scatter(x=proj_tsne[:nb_samples, 0], y=proj_tsne[:nb_samples, 1],
                     color=colors[color_keys][:nb_samples],
                     source=bp.ColumnDataSource({
                         "word": vocabulary[:nb_samples],
                         "sentiment": ws[:nb_samples]
                     })
                     )

    # Hover tool
    hover = plot_we.select(dict(type=HoverTool))
    hover.tooltips = {"Word": "@word", "Sentiment": "@sentiment"}

    save(plot_we, '{}.html'.format(filename))


if __name__ == '__main__':
    embeddings_file = 'embds/500_b_7_9_0.05_10_ns.vec'
    wv, vocabulary = load_embeddings(embeddings_file)
    master_dictionary, md_header, sentiment_categories, stopwords, total_documents = utils.load_masterdictionary(config.SENTIMENT_LEXICON, True, False, True)
    ws = []
    sentiment_categories = sorted(sentiment_categories)

    for w in vocabulary:
        sent_dict = {'negative': 0, 'positive': 0, 'uncertainty': 0, 'litigious': 0, 'constraining': 0, 'strong_modal': 0, 'weak_modal': 0, 'neutral': 0}
        if w in master_dictionary:
            for k, v in master_dictionary[w].sentiment.items():
                if v:
                    sent_dict[k] += 1
        sentiment = sorted(sent_dict.items(), key=lambda x:x[1], reverse=True)[0]
        if sentiment[1] > 0:
            ws.append(sentiment[0])
        else:
            ws.append('neutral')

    nb_samples = len(vocabulary)
    if os.path.exists('embs/tsne_emb.pk'):
        with open('embds/tsne_emb.pk', 'rb') as fp:
            Y = pickle.load(fp)
    else:
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(wv[:nb_samples,:])
        with open('embds/tsne_emb.pk', 'wb') as fp:
            pickle.dump(Y, fp)


    # All word
    colors, color_keys, color_vals = get_colors(ws)
    plot(Y, vocabulary, wv, ws, nb_samples, 'word embeddings', colors, color_keys, 'embds/wemb_all_words')

    # Only sentiment words
    idx_to_remove = set()
    for i, s in enumerate(ws):
        if s == 'neutral':
            idx_to_remove.add(i)
    Y = np.array([y for j, y in enumerate(Y.tolist()) if j not in idx_to_remove])
    vocabulary = [v for j, v in enumerate(vocabulary) if j not in idx_to_remove]
    wv = [v for j, v in enumerate(wv.tolist()) if j not in idx_to_remove]
    ws = [s for j, s in enumerate(ws) if j not in idx_to_remove]

    colors, color_keys, color_vals = get_colors(ws, np.array([sns.xkcd_rgb["purple"], sns.xkcd_rgb["brown"], sns.xkcd_rgb["red"], sns.xkcd_rgb["green"], sns.xkcd_rgb["pink"], sns.xkcd_rgb["orange"]]))
    plot(Y, vocabulary, wv, ws, nb_samples, 'word embeddings', colors, color_keys, 'embds/wemb_only_sent_words')