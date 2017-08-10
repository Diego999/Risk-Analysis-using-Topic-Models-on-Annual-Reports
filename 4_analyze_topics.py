import os
import glob
import re
import stanford_corenlp_pywrapper
import utils
import shutil
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
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
pattern_remove_multiple_dash = re.compile(r'-*')
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


def preprocess_util(data):
    stopwords = set()
    with open(config.STOPWORD_LIST, 'r', encoding='utf-8') as fp:
        for l in fp:
            stopwords.add(l.strip().lower())

    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit, pos, lemma'}, corenlp_jars=[config.SF_NLP_JARS])

    idx = 1
    lemma_to_idx = {'PAD': 0}
    new_data = []
    for item, text in data:
        parsed_data = annotator.parse_doc(text)['sentences']
        assert len(parsed_data) == 1
        parsed_data = parsed_data[0]

        # Lemmatize
        # #lemmas = [l.lower() for l in parsed_data['lemmas']]

        # Lemmatize & remove proper nouns
        #assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
        #lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if 'NP' not in p]

        # Lemmatize & keep only nouns, adjectives and verbs
        assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
        lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if p in ['NN', 'NNS'] or 'VB' in p or 'JJ' in p]

        # Remove stopwords
        lemmas = [l for l in lemmas if l not in stopwords]

        # Construct dictionnary
        for lemma in lemmas:
            if lemma not in lemma_to_idx:
                lemma_to_idx[lemma] = idx
                idx += 1

        # Replace lemma by their corresponding indices
        lemmas_idx = [lemma_to_idx[lemma] for lemma in lemmas]

        new_data.append((item, lemmas_idx))

    idx_to_lemma = {v: k for k, v in lemma_to_idx.items()}

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


def tfidf(data, min_threshold=0.0, max_threshold=1.0, idx_to_lemma=None):
    max_lemmas = max([len(lemmas) for item, lemmas in data])
    X = lil_matrix((len(data), max_lemmas))

    # Fill the matrix X with idx of the lemmas of each report
    for d in range(len(data)):
        for i in range(len(data[d][1])):
            X[d, i] = data[d][1][i]

    transformer = TfidfTransformer(smooth_idf=True)
    tfidf_results = transformer.fit_transform(X).toarray()

    # Filter lemmas by their tf-idf
    new_data = []
    with open('lemmas.txt', 'w') as fp: # To be removed
        for i in range(len(data)):
            new_lemmas = []
            for lemma, tfidf in zip(data[i][1], tfidf_results[i]):
                if min_threshold <= tfidf <= max_threshold:
                    fp.write(idx_to_lemma[lemma] + '\t' + str(tfidf) + '\n') # To be removed
                    new_lemmas.append((lemma, tfidf))
            new_data.append((data[i][0], new_lemmas))

    return new_data


def prepare_data(section, data, idx_to_lemma):
    folder_input = section + config.SUFFIX_INPUT_DATA

    if config.FORCE_PREPROCESSING and os.path.isdir(folder_input):
        shutil.rmtree(folder_input)

    if not os.path.isdir(folder_input):
        os.mkdir(folder_input)

    for item, lemmas in data:
        item_filename = os.path.join(folder_input, item[item.rfind('/')+1:])
        with open(item_filename, 'w', encoding='utf-8') as fp:
            fp.write(' '.join([idx_to_lemma[idx_lemma] for idx_lemma, tfidf in lemmas]))


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]

    for section in sections_to_analyze:
        data = load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = preprocess(section, data)
        data = tfidf(data, min_threshold=0.0, max_threshold=1.0, idx_to_lemma=idx_to_lemma)
        prepare_data(section, data, idx_to_lemma)