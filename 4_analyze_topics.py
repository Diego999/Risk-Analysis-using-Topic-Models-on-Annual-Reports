import os
import glob
import re
import stanford_corenlp_pywrapper
import utils
import logging
import numpy
import collections
from multiprocessing import Process, Manager
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, HdpModel, LdaMulticore
from gensim.models.wrappers import LdaMallet
config = __import__('0_config')


try:
    import pyLDAvis.gensim
    CAN_VISUALIZE = True
except ImportError:
    CAN_VISUALIZE = False


def read_section(section):
    if not os.path.isdir(section):
        print('ERR: {} does not exists !'.format(section))
        exit(-1)

    return glob.glob(section + '/*.' + config.EXTENSION_10K_REPORT)


def remove_header_and_multiple_lines(buffer):
    text = ''.join(buffer[1:])
    text = re.sub(r'\n+', '\n', text).strip()
    return text.split('\n')


def content_relevant(buffer):
    keywords = ['smaller reporting company', 'pages', 'discussion and analysis of financial condition']
    text = ''.join(buffer).lower()

    if len(text) < 400:
        return False

    if len(buffer) < 50 and len(text) < 5000:
        for kw in keywords:
            if kw in text:
                return False

    return True


pattern_remove_multiple_space = re.compile(r'\s+')
pattern_remove_new_lines = re.compile(r'\n+')
pattern_non_char_digit_hash_perc_dash = re.compile(r'[^a-z0-9#%\' -]+')
pattern_remove_multiple_dash = re.compile(r'-+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
pattern_multiple_hash = re.compile(r'#+')
def clean(buffer):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_non_char_digit_hash_perc_dash, ' ', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    text = re.sub(pattern_remove_multiple_dash, '-', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    return text.strip()


def load_and_clean_data(section):
    cleaned_data_file = section + config.SUFFIX_CLEAN_DATA
    cleaned_data = []
    if config.FORCE_PREPROCESSING or not os.path.exists(cleaned_data_file):
        items = read_section(section)
        for item in items:
            buffer = []
            with open(item, 'r', encoding='utf-8') as fp:
                for l in fp:
                    buffer.append(l)

            buffer = remove_header_and_multiple_lines(buffer)
            if content_relevant(buffer):
                cleaned_data.append((item, clean(buffer)))

        print('{}: Keep {} over {} ({:.2f}%)'.format(section[section.rfind('/') + 1:], len(cleaned_data), len(items), float(len(cleaned_data)) / len(items) * 100.0))

        with open(cleaned_data_file, 'w', encoding='utf-8') as fp:
            for i, l in cleaned_data:
                fp.write(i + '\t' + l + '\n')
    else:
        with open(cleaned_data_file, 'r', encoding='utf-8') as fp:
            for l in fp:
                i, l = l.split('\t')
                cleaned_data.append((i, l.strip()))

    return cleaned_data


def get_lemmas(parsed_data):
    assert len(parsed_data) == 1
    parsed_data = parsed_data[0]

    # Lemmatize
    lemmas = [l.lower() for l in parsed_data['lemmas']]

    # Lemmatize & remove proper nouns
    #assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
    #lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if 'NP' not in p]

    # Lemmatize & keep only nouns, adjectives and verbs
    #assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
    #lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if p in ['NN', 'NNS'] or 'VB' in p or 'JJ' in p]

    return lemmas


def preprocess_util_thread(data, storage, pid):
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit, pos, lemma'}, corenlp_jars=[config.SF_NLP_JARS])

    new_data = []
    for item, text in data:
        parsed_data = annotator.parse_doc(text)['sentences']
        lemmas = get_lemmas(parsed_data)
        new_data.append((item, lemmas))

    storage[pid] = new_data


def construct_lemma_dict_and_transform_lemma_to_idx(lemmas, lemma_to_idx, idx):
    # Construct dictionnary
    for lemma in lemmas:
        if lemma not in lemma_to_idx:
            lemma_to_idx[lemma] = idx
            idx += 1

    # Replace lemma by their corresponding indices
    lemmas_idx = [lemma_to_idx[lemma] for lemma in lemmas]

    return lemmas_idx, lemma_to_idx, idx


def add_bigrams_trigrams(data):
    docs = [x[1] for x in data]

    bigrams = Phrases(docs, min_count=config.MIN_FREQ)
    for idx in range(len(docs)):
        for lemma in bigrams[docs[idx]]:
            if '_' in lemma: # a_b
                docs[idx].append(lemma)

    # Trigrams are computer by reiterating with bigrams
    trigrams = Phrases(bigrams, min_count=config.MIN_FREQ)
    for idx in range(len(docs)):
        for lemma in trigrams[docs[idx]]:
            if lemma.count('_') > 1: # a_b_c
                docs[idx].append(lemma)

    for i in range(len(data)):
        data[i] = (data[i][0], docs[i])

    return data


def remove_stopwords(data):
    stopwords = set()
    with open(config.STOPWORD_LIST, 'r', encoding='utf-8') as fp:
        for l in fp:
            stopwords.add(l.strip().lower())

    # Remove stopwords
    new_data = []
    for item, lemmas in data:
        new_data.append((item, [l for l in lemmas if l not in stopwords and (len(l) > 1 and l != '#')]))

    return new_data


def transform_bow(data):
    new_data = []
    idx = 1
    lemma_to_idx = {'PAD': 0}
    for item, lemmas in data:
        lemmas_idx, lemma_to_idx, idx = construct_lemma_dict_and_transform_lemma_to_idx(lemmas, lemma_to_idx, idx)
        new_data.append((item, lemmas_idx))
    idx_to_lemma = {v: k for k, v in lemma_to_idx.items()}

    return new_data, lemma_to_idx, idx_to_lemma


def preprocess_util(data):
    manager = Manager()
    if config.MULTITHREADING:
        data = utils.chunks(data, 1 + int(len(data) / config.NUM_CORES))
        final_data = manager.list([[]] * config.NUM_CORES)

        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=preprocess_util_thread, args=(data[i], final_data, i)))
            procs[-1].start()

        for p in procs:
            p.join()

    else:
        final_data = [None]
        preprocess_util_thread(data, final_data, 0)
    new_data = sum(final_data, [])

    new_data = add_bigrams_trigrams(new_data)
    new_data = remove_stopwords(new_data)
    new_data, lemma_to_idx, idx_to_lemma = transform_bow(new_data)

    return new_data, lemma_to_idx, idx_to_lemma


def preprocess(section, data):
    preprocessed_data_file = section + config.SUFFIX_PREPROCESSED_DATA
    dict_lemma_idx_file = section + config.DICT_LEMMA_IDX
    dict_idx_lemma_file = section + config.DICT_IDX_LEMMA

    preprocessed_data = None
    lemma_to_idx = None
    idx_to_lemma = None
    if config.FORCE_PREPROCESSING or not os.path.exists(preprocessed_data_file) or not os.path.exists(dict_lemma_idx_file) or not os.path.exists(dict_idx_lemma_file):
        preprocessed_data, lemma_to_idx, idx_to_lemma = preprocess_util(data)

        # Save files
        utils.save_pickle(preprocessed_data, preprocessed_data_file)
        utils.save_pickle(lemma_to_idx, dict_lemma_idx_file)
        utils.save_pickle(idx_to_lemma, dict_idx_lemma_file)
    else:
        # Load files
        preprocessed_data = utils.load_pickle(preprocessed_data_file)
        lemma_to_idx = utils.load_pickle(dict_lemma_idx_file)
        idx_to_lemma = utils.load_pickle(dict_idx_lemma_file)

    return (preprocessed_data, lemma_to_idx, idx_to_lemma)


def sort_by_year_annual_report(item):
    return utils.year_annual_report_comparator(utils.extract_year_from_filename_annual_report(item))


def preprocessing_topic(data, idx_to_lemma):
    for i in range(0, len(data)):
        data[i] = (data[i][0], [idx_to_lemma[idx] for idx in data[i][1]])

    # Sort data & create time slices to use dynamic topic modeling and analyze evolution of topics during the time (93 94 ... 2015 2016 2017 ...)
    data = sorted(data, key=lambda x: sort_by_year_annual_report(x[0]))
    data_years = [utils.extract_year_from_filename_annual_report(item) for item, text in data]
    time_slices = [freq for year, freq in sorted(collections.Counter(data_years).items(), key=lambda x: utils.year_annual_report_comparator(x[0]))]
    assert sum(time_slices) == len(data)

    texts = [x[1] for x in data]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=config.MIN_FREQ, no_above=config.MAX_DOC_RATIO)
    corpus = [dictionary.doc2bow(x[1]) for x in data]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return corpus, dictionary, texts


def visualize(model, corpus, dictionary):
    if CAN_VISUALIZE:
        prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        pyLDAvis.show(prepared)


def train_LDA(corpus, dictionary, texts):
    # Set training parameters.
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = 1

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # Gensim LDA
    #lda_model, coherence_c_v, coherence_u_mass = train_lda_model(corpus, dictionary, chunksize, eval_every, id2word, iterations, num_topics, passes, texts)
    #print(coherence_u_mass.get_coherence(), coherence_c_v.get_coherence())
    #visualize(lda_model, corpus, dictionary)

    # LDA Multicore
    lda_model, coherence_c_v, coherence_u_mass = train_lda_model_multicores(corpus, dictionary, chunksize, eval_every, id2word, iterations, num_topics, passes, texts, workers=int(config.NUM_CORES/2)-1)
    print(coherence_u_mass.get_coherence(), coherence_c_v.get_coherence())
    visualize(lda_model, corpus, dictionary)
def train_lda_model(corpus, dictionary, chunksize, eval_every, id2word, iterations, num_topics, passes, texts):
    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
    coherence_u_mass = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_c_v = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return model, coherence_c_v, coherence_u_mass


def train_lda_model_multicores(corpus, dictionary, chunksize, eval_every, id2word, iterations, num_topics, passes, texts, workers):
    model = LdaMulticore(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='symmetric', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every, workers=workers)
    coherence_u_mass = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_c_v = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return model, coherence_c_v, coherence_u_mass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(0)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        data = load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = preprocess(section, data)

        corpus, dictionary, texts = preprocessing_topic(data, idx_to_lemma)
        train_LDA(corpus, dictionary, texts)