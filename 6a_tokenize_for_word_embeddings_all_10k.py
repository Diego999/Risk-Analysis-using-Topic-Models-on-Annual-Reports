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
pattern_url = re.compile(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))")
def clean(buffer, KEYWORDS_TO_DETECT, MIN_LENGTH_EMPTY, MIN_LENGTH_KEYWORDS, MIN_LINES_KEYWORDS, REMOVE_START_WORDS):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_remove_quote, '', text)
    text = re.sub(pattern_remove_quote_2, '', text)
    text = re.sub(pattern_remove_quote_3, '', text)
    text = re.sub(pattern_remove_quote_4, '', text)
    text = re.sub(pattern_multiple_equal_underline, '', text)
    text = re.sub(pattern_url, ' URL ', text)
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


def load_and_clean_data_process(items, pid):
    cleaned_data = []
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit'}, corenlp_jars=[config.SF_NLP_JARS])
    for i, item in enumerate(items):
        buffer = []
        with open(item, 'r', encoding='utf-8') as fp:
            for l in fp:
                buffer.append(l)

        if len(buffer) > 200:
            buffer = buffer[200:]
            if len(buffer) > 250:
                buffer = buffer[:-250]

            buffer = [l for l in buffer if len(l.strip()) > 75 and not l.startswith('vrnt') and not l.startswith('us-gaap') and not l.startswith('uan') and not l.startswith('xbrli') and not l.startswith('ccitii')]
            buffer = analyze_topics_static.remove_header_and_multiple_lines(buffer)
            cleaned_data = nlp_process(clean(buffer, None, None, None, None, None), annotator)

        if len(cleaned_data) > 0:
            item = item[item.rfind('/')+1:]
            with open(config.DATA_TEMP_FOLDER + '/' + item, 'w', encoding='utf-8') as fp:
                for l in cleaned_data.split(SENTENCE_DELIMITER):
                    fp.write(l + '\n')


def load_and_clean_data(reports):
    if config.MULTITHREADING:
        items_chunk = utils.chunks(reports, 1 + int(len(reports) / config.NUM_CORES))
        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=load_and_clean_data_process, args=(items_chunk[i], i)))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        load_and_clean_data_process(reports)


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    path = os.path.join(config.DATA_TEMP_FOLDER)
    if not os.path.exists(path):
        os.makedirs(path)

    reports = glob.glob(config.DATA_AR_FOLDER + '/*/*.' + config.EXTENSION_CLEAN_PREPROCESSING)
    data = load_and_clean_data(reports)

    cleaned_reports = glob.glob(config.DATA_TEMP_FOLDER + '/*.' + config.EXTENSION_CLEAN_PREPROCESSING)
    output = os.path.join(config.DATA_AR_FOLDER, 'all_10k_for_we.txt')
    with open(output, 'w', encoding='utf-8') as fp_o:
        for r in cleaned_reports:
            with open(r, 'r', encoding='utf-8') as fp_i:
                for l in fp_i:
                    fp_o.write(l.strip() + '\n')
    

