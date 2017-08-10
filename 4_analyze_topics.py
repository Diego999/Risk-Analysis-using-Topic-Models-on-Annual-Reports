import os
import glob
import re
import stanford_corenlp_pywrapper
import utils
import shutil
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from multiprocessing import Process, Manager
from gensim.models import Phrases
from gensim.corpora import Dictionary
config = __import__('0_config')


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
    # assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
    # lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if 'NP' not in p]

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


def add_bigrams(data):
    docs = [x[1] for x in data]
    bigram = Phrases(docs, min_count=config.MIN_FREQ)
    for idx in range(len(docs)):
        for lemma in bigram[docs[idx]]:
            if '_' in lemma:
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

    new_data = add_bigrams(new_data)
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


def preprocessing_topic(data, idx_to_lemma):
    for i in range(0, len(data)):
        data[i] = (data[i][0], [idx_to_lemma[idx] for idx in data[i][1]])

    dictionary = Dictionary([x[1] for x in data])
    dictionary.filter_extremes(no_below=config.MIN_FREQ, no_above=config.MAX_DOC_RATIO)
    corpus = [dictionary.doc2bow(x[1]) for x in data]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]

    for section in sections_to_analyze:
        data = load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = preprocess(section, data)
        preprocessing_topic(data, idx_to_lemma)