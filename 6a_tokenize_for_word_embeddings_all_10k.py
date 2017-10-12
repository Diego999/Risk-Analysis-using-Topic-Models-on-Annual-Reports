import os
import logging
import numpy
import random
import re
import utils
import stanford_corenlp_pywrapper
import glob
from multiprocessing import Process, Manager
analyze_topics_static = __import__('4a_analyze_topics_static')
infer_topics = __import__('4d_infer_topics')
config = __import__('0_config')


SENTENCE_DELIMITER = ' <DELIMITER> '
pattern_remove_multiple_space = re.compile(r'\s+')
pattern_remove_new_lines = re.compile(r'\n+')
pattern_non_char_digit_hash_perc_dash = re.compile(r"[^a-z0-9$#%' .-]+$")
pattern_remove_quote = re.compile(r"('')+")
pattern_remove_quote_2 = re.compile(r'"+')
pattern_remove_quote_3 = re.compile(r"“+")
pattern_remove_quote_4 = re.compile(r"”+")
pattern_remove_multiple_dash = re.compile(r'-+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
pattern_multiple_hash = re.compile(r'#+')
pattern_multiple_equal_underline = re.compile(r'[=_]+')
pattern_dot_among_hashes = re.compile(r'#+ *. *#+')
pattern_comma_among_hashes = re.compile(r'#+ *, *#+')
remove_numbers_redundancy = re.compile(r'[% -]*#+[% -]*')
remove_numbers_redundancy_2 = re.compile(r'\( *# *\)')
remove_numbers_redundancy_3 = re.compile(r'#a')
remove_numbers_redundancy_4 = re.compile(r'#[,. ]+#')
remove_multiple_dots = re.compile(r'\. *\.+')
remove_multiple_dashes = re.compile(r'([-] )+')
remove_multiple_dollars = re.compile(r'([$] [-])+')
remove_multiple_numbers = re.compile(r'( NUM )+')
def clean(buffer, KEYWORDS_TO_DETECT, MIN_LENGTH_EMPTY, MIN_LENGTH_KEYWORDS, MIN_LINES_KEYWORDS, REMOVE_START_WORDS):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_remove_quote, '', text)
    text = re.sub(pattern_remove_quote_2, '', text)
    text = re.sub(pattern_remove_quote_3, '', text)
    text = re.sub(pattern_remove_quote_4, '', text)
    text = re.sub(pattern_multiple_equal_underline, '', text)
    text = re.sub(pattern_non_char_digit_hash_perc_dash, ' ', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    text = re.sub(pattern_remove_multiple_dash, '-', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)

    for vals in [' ten ', ' tens ', ' hundred ', ' hundreds ', ' million ', ' millions ', ' billion ', ' billions ']:
        text = text.replace(vals, '#')
        text = text.replace(vals[:-1] + '.', '#')
        text = text.replace(vals[:-1] + ',', '#')
        text = text.replace(vals[:-1], '#')
    for curr in ['$', '€']:
        text = text.replace(curr, ' CUR ')
    for curr in [' dollar ', ' dollars ', ' euro ', ' euros ', ' eur ']:
        text = text.replace(curr, ' CUR ')
        text = text.replace(curr[:-1] + '.', ' CUR ')
        text = text.replace(curr[:-1] + ',', ' CUR ')
        text = text.replace(curr[:-1], ' CUR ')

    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    text = re.sub(pattern_dot_among_hashes, '#', text)
    text = re.sub(pattern_comma_among_hashes, '#', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    text = re.sub(remove_numbers_redundancy_2, '#', text)
    text = re.sub(remove_numbers_redundancy_3, '#', text)
    text = re.sub(remove_numbers_redundancy_4, '#', text)
    text = re.sub(remove_numbers_redundancy, ' NUM ', text)
    text = re.sub(remove_multiple_dots, '.', text)
    text = re.sub(remove_multiple_dashes, '', text)
    text = re.sub(remove_multiple_dollars, '$', text)
    text = re.sub(remove_multiple_numbers, ' NUM ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = text.strip()

    return text.strip()


def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token


def nlp_process(text, annotator):
    tokens_sent = []
    for parsed_data in annotator.parse_doc(text)['sentences']:
        tokens_sent.append(' '.join([format_token(t) for t in parsed_data['tokens']]))

    return SENTENCE_DELIMITER.join(tokens_sent)


def load_and_clean_data_process(items, output_folder, storage, pid):
    cleaned_data = []
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit'}, corenlp_jars=[config.SF_NLP_JARS])
    for i, item in enumerate(items):
        buffer = []
        with open(item, 'r', encoding='utf-8') as fp:
            for l in fp:
                buffer.append(l)

        if len(buffer) > (200 + 250):
            buffer = buffer[200:-250]
            buffer = analyze_topics_static.remove_header_and_multiple_lines(buffer)
            cleaned_data.append((item, nlp_process(clean(buffer, None, None, None, None, None), annotator)))

    storage[pid] = cleaned_data


def load_and_clean_data(output_folder, reports):
    cleaned_data_file = output_folder + '/all_10k_' + config.SUFFIX_CLEAN_DATA
    cleaned_data = []
    if config.FORCE_PREPROCESSING or not os.path.exists(cleaned_data_file):
        manager = Manager()
        if config.MULTITHREADING:
            items_chunk = utils.chunks(reports, 1 + int(len(reports) / config.NUM_CORES))
            final_data = manager.list([[]] * config.NUM_CORES)

            procs = []
            for i in range(config.NUM_CORES):
                procs.append(Process(target=load_and_clean_data_process, args=(items_chunk[i], output_folder, final_data, i)))
                procs[-1].start()

            for p in procs:
                p.join()
        else:
            final_data = [None]
            load_and_clean_data_process(reports, output_folder, final_data, 0)
        cleaned_data = sum(final_data, [])

        with open(cleaned_data_file, 'w', encoding='utf-8') as fp:
            for i, l in cleaned_data:
                fp.write(i + '\t' + l + '\n')
    else:
        with open(cleaned_data_file, 'r', encoding='utf-8') as fp:
            for l in fp:
                i, l = l.split('\t')
                cleaned_data.append((i, l.strip()))

    return cleaned_data


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    reports = glob.glob(config.DATA_AR_FOLDER + '/*/*.' + config.EXTENSION_CLEAN_PREPROCESSING)

    data = load_and_clean_data(config.DATA_FOLDER, reports)
    output = config.DATA_FOLDER + '/all_10k_' + config.SUFFIX_CLEAN_DATA + config.SUFFIX_PREPROCESSED_DATA_FOR_WE.replace('pkl', 'txt')
    with open(output, 'w', encoding='utf-8') as fp:
        for item, text in data:
            for sent in text.split(SENTENCE_DELIMITER):
                fp.write(sent + '\n')

