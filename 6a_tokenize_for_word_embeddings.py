import os
import logging
import numpy
import random
import re
import utils
import collections
import stanford_corenlp_pywrapper
import pickle
from gensim.models import Phrases
from gensim.corpora import Dictionary
from multiprocessing import Process, Manager
analyze_topics_static = __import__('4a_analyze_topics_static')
infer_topics = __import__('4d_infer_topics')
config = __import__('0_config')


pattern_remove_multiple_space = re.compile(r'\s+')
pattern_remove_new_lines = re.compile(r'\n+')
pattern_non_char_digit_hash_perc_dash = re.compile(r'[^a-z0-9#%\' -.\']+') # Override we want to keep sentence
pattern_remove_multiple_dash = re.compile(r'-+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
pattern_multiple_hash = re.compile(r'#+')
pattern_dot_among_hashes = re.compile(r'#+ *. *#+') # Added
pattern_comma_among_hashes = re.compile(r'#+ *, *#+') # Added
remove_numbers_redundancy = re.compile(r'[%$ -]*#+[%$ -]*')
remove_numbers_redundancy_2 = re.compile(r'\( *# *\)')
remove_numbers_redundancy_3 = re.compile(r'#a')
remove_numbers_redundancy_4 = re.compile(r'#[,. ]+#')
remove_multiple_dots = re.compile(r'\. *\.+')
remove_multiple_dashes = re.compile(r'([-] )+')
remove_multiple_dollars = re.compile(r'([$] [-])+')
remove_multiple_numbers = re.compile(r'( _NUM_ )+')
def clean(buffer, KEYWORDS_TO_DETECT, MIN_LENGTH_EMPTY, MIN_LENGTH_KEYWORDS, MIN_LINES_KEYWORDS, REMOVE_START_WORDS):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_non_char_digit_hash_perc_dash, ' ', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    text = re.sub(pattern_remove_multiple_dash, '-', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    text = re.sub(pattern_dot_among_hashes, '#', text) # Added
    text = re.sub(pattern_comma_among_hashes, '#', text) # Added
    text = re.sub(pattern_multiple_hash, '#', text) # Added
    text = re.sub(remove_numbers_redundancy_2, '#', text) # Added
    text = re.sub(remove_numbers_redundancy_3, '#', text) # Added
    text = re.sub(remove_numbers_redundancy_4, '#', text) # Added
    text = re.sub(remove_numbers_redundancy, ' _NUM_ ', text) # Added
    text = re.sub(remove_multiple_dots, '.', text) # Added
    text = re.sub(remove_multiple_dashes, '', text) # Added
    text = re.sub(remove_multiple_dollars, '$', text) # Added
    text = re.sub(remove_multiple_numbers, ' _NUM_ ', text) # Added
    text = re.sub(pattern_remove_multiple_space, ' ', text) # Added
    text = text.strip()

    start_over = True
    while start_over:
        start_over = False
        for remove_start_words in REMOVE_START_WORDS:
            cond_valid = text.startswith(remove_start_words)
            if not cond_valid:
                remove_start_words = remove_start_words.replace('#', '_NUM_ ').replace('.', '')
                cond_valid = text.startswith(remove_start_words)
            if cond_valid:
                text = text[len(remove_start_words) + 1:]
                while text[0] in [' ', '-', '#']:
                    text = text[1:]
                start_over = True

    return text.strip()


def load_and_clean_data_process(items, section, storage, pid):
    cleaned_data = []
    for i, item in enumerate(items):
        buffer = []
        with open(item, 'r', encoding='utf-8') as fp:
            for l in fp:
                buffer.append(l)

        buffer = analyze_topics_static.remove_header_and_multiple_lines(buffer)
        if analyze_topics_static.content_relevant(buffer, *config.CLEAN_PARAMETERS[section]):
            cleaned_data.append((item, clean(buffer, *config.CLEAN_PARAMETERS[section]))) #Override clean function because we want to keep the '.'

    storage[pid] = cleaned_data


def load_and_clean_data(section):
    cleaned_data_file = section + config.SUFFIX_CLEAN_DATA + config.SUFFIX_PREPROCESSED_DATA_FOR_WE.replace('pkl', 'txt')
    cleaned_data = []
    if config.FORCE_PREPROCESSING or not os.path.exists(cleaned_data_file):
        items = analyze_topics_static.read_section(section)
        manager = Manager()
        if config.MULTITHREADING:
            items_chunk = utils.chunks(items, 1 + int(len(items) / config.NUM_CORES))
            final_data = manager.list([[]] * config.NUM_CORES)

            procs = []
            for i in range(config.NUM_CORES):
                procs.append(Process(target=load_and_clean_data_process, args=(items_chunk[i], section, final_data, i)))
                procs[-1].start()

            for p in procs:
                p.join()
        else:
            final_data = [None]
            load_and_clean_data_process(items, section, final_data, 0)
        cleaned_data = sum(final_data, [])

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


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        path = os.path.join(config.OUTPUT_FOLDER_WORD_EMB, section[section.rfind('/')+1:])
        if not os.path.exists(path):
            os.makedirs(path)

        data = load_and_clean_data(section)  # Override in order to keep '.' to distinguish sentences
