import os
import glob
import re
import stanford_corenlp_pywrapper
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
pattern_non_char_digit_hash_perc = re.compile(r'[^a-z0-9#%\' ]+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
pattern_multiple_hash = re.compile(r'#+')
def clean(buffer):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_non_char_digit_hash_perc, ' ', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    return text.strip()


def load_and_clean_data(section):
    final_file = section + '.' + config.EXTENSION_10K_REPORT
    filtered_items = []
    if not os.path.exists(final_file):
        items = read_section(section)

        for item in items:
            buffer = []
            with open(item, 'r', encoding='utf-8') as fp:
                for l in fp:
                    buffer.append(l)

            buffer = remove_header_and_multiple_lines(buffer)
            if content_relevant(buffer):
                filtered_items.append((item, clean(buffer)))

        print('{}: Keep {} over {} ({:.2f}%)'.format(section[section.rfind('/') + 1:], len(filtered_items), len(items), float(len(filtered_items)) / len(items) * 100.0))

        with open(final_file, 'w', encoding='utf-8') as fp:
            for i, l in filtered_items:
                fp.write(i + '\t' + l + '\n')
    else:
        with open(final_file, 'r', encoding='utf-8') as fp:
            for l in fp:
                i, l = l.split('\t')
                filtered_items.append((i, l.strip()))

    return filtered_items


def lemmatize(data):
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit, pos, lemma'}, corenlp_jars=[config.SF_NLP_JARS])

    new_data = []
    for item, text in data:
        parsed_data = annotator.parse_doc(text)['sentences']
        assert len(parsed_data) == 1
        parsed_data = parsed_data[0]

        lemmas = [l.lower() for l in parsed_data['lemmas']]

        new_data.append((item, lemmas))

    return new_data


def remove_stopwords(data):
    stopwords = set()
    with open(config.STOPWORD_LIST, 'r', encoding='utf-8') as fp:
        for l in fp:
            stopwords.add(l.strip().lower())

    new_data = []
    for item, lemmas in data:
        new_data.append((item, [l for l in lemmas if l not in stopwords]))

    return new_data


def tfidf(data, min_threshold=0.0, max_threshold=1.0):
    idx = 1
    lemma_to_idx = {'PAD': 0}
    for item, lemmas in data:
        for lemma in lemmas:
            if lemma not in lemma_to_idx:
                lemma_to_idx[lemma] = idx
                idx += 1
    idx_to_lemma = {v: k for k, v in lemma_to_idx.items()}

    max_lemmas = max([len(lemmas) for item, lemmas in data])
    X = lil_matrix((len(data), max_lemmas))

    for d in range(len(data)):
        for i in range(len(data[d][1])):
            X[d, i] = lemma_to_idx[data[d][1][i]]

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_results = transformer.fit_transform(X).toarray()

    new_data = []
    for i in range(len(data)):
        new_lemmas = []
        for lemma, tfidf in zip(data[i][1], tfidf_results[i]):
            if min_threshold <= tfidf <= max_threshold:
                new_lemmas.append(lemma)
        new_data.append((data[i][0], new_lemmas))

    return new_data


def preprocess(data):
    new_data = lemmatize(data)
    new_data = remove_stopwords(new_data)
    new_data = tfidf(new_data)
    return new_data


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]

    for section in sections_to_analyze:
        data = load_and_clean_data(section)

    data = preprocess(data)
