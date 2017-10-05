import utils
import os
import glob
import numpy as np
import math
analyze_topics_static = __import__('4a_analyze_topics_static')
infer_topics = __import__('4d_infer_topics')
config = __import__('0_config')


def get_topic_per_documents(section, data_):
    data, lemma_to_idx, idx_to_lemma = analyze_topics_static.preprocess(section, data_)
    corpus, dictionary, texts, time_slices, data = analyze_topics_static.preprocessing_topic(data, idx_to_lemma)

    model_filepath = config.TRAIN_PARAMETERS[section][1]
    assert os.path.exists(model_filepath)
    model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

    num_topics = config.TRAIN_PARAMETERS[section][0]
    return infer_topics.get_document_topic_distribution(num_topics, data, corpus, model, config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.DOCS_EXTENSION))


def get_data_from_pickles(filename, data_filenames):
    if filename not in data_filenames:
        return None

    data = utils.load_pickle(data_filenames[filename])
    # Remove duplicate
    for i in range(0, len(data)):
        for j in range(1, len(data)):
            if data[i] == data[j]:
                data[j] = None
    data = [x for x in data if x is not None]
    assert len(data) == 1
    data = data[0]

    return data


def compute_ln_volatility(stocks):
    if stocks is None or len(stocks) == 0:
        return float('nan')

    all_Pts = [x[1] for x in sorted(stocks.items(), key=lambda x:x[0]) if not math.isnan(x[1])] # Sort by date
    # There might be a "nan" on single values, we should remove them to avoid having a bad computation !
    Pt = np.array(all_Pts[1:])
    Pt_1 = np.array(all_Pts[:-1])
    rt = Pt/Pt_1 - 1.0

    # Warning: depending the paper, rt = ln(Pt) - ln(Pt_1)
    return np.log(np.std(rt)) # ln(std(rt)) where rt = Pt - P_(t-1)


# Return ROF (*100 to have %)
def compute_return_on_equity(filename, ni_seq):
    if ni_seq is None or len(ni_seq) == 0:
        return float('nan'), float('nan'), float('nan')

    ni_seq_per_year = {k.split('-')[0][-2:]:v for k, v in ni_seq.items()}
    file_year = filename.split('_')[1].split('-')[1]
    vals = ni_seq_per_year[file_year]
    ni = vals['ni']
    seq = vals['seq'] if not math.isnan(vals['seq']) else vals['teq']
    roe = float('nan')
    if not math.isnan(seq) and not math.isnan(ni):
        roe = ni/seq

    return ni, seq, roe


stocks = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_STOCKS_FOLDER))}
ni_seqs = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_NI_SEQ_FOLDER))}
sec_inds = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_SEC_IND_FOLDER))}
if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)
        topics = {x[0][x[0].rfind('/')+1:-4]:x[1] for x in get_topic_per_documents(section, data)}

        data = {x[0].split('/')[-1].split('.')[0]:{'text': x[1]} for x in data}
        # add topics
        for file, attributes in data.items():
            data[file]['topics'] = topics[file]

        # add ln volatility and return on equity
        for file, attributes in data.items():
            current_stocks = get_data_from_pickles(file, stocks)
            data[file]['volatility'] = compute_ln_volatility(current_stocks)
            ni_seq = get_data_from_pickles(file, ni_seqs)
            data[file]['return_on_equity'] = compute_return_on_equity(file, ni_seq)