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
    text = text.strip()

    for remove_start_words in REMOVE_START_WORDS:
        if text.startswith(remove_start_words):
            text = text[len(remove_start_words) + 1:]
            while text[0] in [' ', '-', '#']:
                text = text[1:]

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
    cleaned_data_file = section + config.SUFFIX_CLEAN_DATA + config.SUFFIX_PREPROCESSED_DATA_FOR_SENT.replace('pkl', 'txt')
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


def preprocess_util_thread(data, storage, pid):
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit, pos, lemma'}, corenlp_jars=[config.SF_NLP_JARS])

    new_data = []
    for item, text in data:
        parsed_data = annotator.parse_doc(text)['sentences']
        lemmas = []
        tokens = []
        for p in parsed_data:
            lemmas.append(analyze_topics_static.get_lemmas([p]))
            tokens.append(analyze_topics_static.get_tokens([p]))
        new_data.append((item, tokens, lemmas))

    storage[pid] = new_data


def add_bigrams_trigrams(data):
    docs_sents = [lemmas for item, tokens, lemmas in data]
    docs = [sum(d, []) for d in docs_sents]

    # Learn in the overall corpus
    print('bi')
    bigrams = Phrases(docs, min_count=config.MIN_FREQ_BI) # a b -> a_b
    print('tri')
    trigrams = Phrases([bigrams[docs[x]] for x in range(len(docs))], min_count=config.MIN_FREQ_TRI) # a b c -> a_b_c
    print('quadri')
    quadrigrams = Phrases([trigrams[docs[x]] for x in range(len(docs))], min_count=config.MIN_FREQ_QUA) # a b c d -> a_b_c_d

    # Infer n-grams per sentence
    for idx in range(len(data)):
        data[idx] = (data[idx][0], data[idx][1], [quadrigrams[docs] for docs in docs_sents[idx]])

    return data


def remove_stopwords(data):
    stopwords = set()
    with open(config.STOPWORD_LIST, 'r', encoding='utf-8') as fp:
        for l in fp:
            stopwords.add(l.strip().lower())

    # Remove stopwords
    new_data = []
    for item, tokens_sent, lemmas_sent in data:
        new_data.append((item, tokens_sent, [[l for l in lemmas if l not in stopwords and (len(l) > 1 and l != '#')] for lemmas in lemmas_sent]))

    return new_data


def transform_bow(data, lemma_to_idx, idx_to_lemma):
    new_data = []
    for item, tokens_sent, lemmas_sent in data:
        lemmas_idx_sent = []
        for lemmas in lemmas_sent:
            lemmas_idx = []
            for lemma in lemmas:
                if lemma in lemma_to_idx:
                    lemmas_idx.append(lemma_to_idx[lemma])
            lemmas_idx_sent.append(lemmas_idx)
        new_data.append((item, tokens_sent, lemmas_idx_sent))

    return new_data


def preprocess_util(data, lemma_to_idx, idx_to_lemma):
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
    new_data = transform_bow(new_data, lemma_to_idx, idx_to_lemma)

    return new_data


def preprocess(section, data):
    preprocessed_data_file = section + config.SUFFIX_PREPROCESSED_DATA + config.SUFFIX_PREPROCESSED_DATA_FOR_SENT
    # We need to keep the same dict file which has been served for topic modeling
    lemma_to_idx = utils.load_pickle(section + config.DICT_LEMMA_IDX)
    idx_to_lemma = utils.load_pickle(section + config.DICT_IDX_LEMMA)

    preprocessed_data = None
    if config.FORCE_PREPROCESSING or not os.path.exists(preprocessed_data_file):
        preprocessed_data = preprocess_util(data, lemma_to_idx, idx_to_lemma)

        # Save files
        utils.save_pickle(preprocessed_data, preprocessed_data_file)
    else:
        # Load files
        preprocessed_data = utils.load_pickle(preprocessed_data_file)

    return (preprocessed_data, lemma_to_idx, idx_to_lemma)


def preprocessing_topic(data, idx_to_lemma):
    for i in range(0, len(data)):
        data[i] = (data[i][0], data[i][1], [[idx_to_lemma[idx_] for idx_ in idx] for idx in data[i][2]])

    # Sort data & create time slices to use dynamic topic modeling and analyze evolution of topics during the time (93 94 ... 2015 2016 2017 ...)
    data = sorted(data, key=lambda x: analyze_topics_static.sort_by_year_annual_report(x[0]))
    data_years = [utils.extract_year_from_filename_annual_report(item) for item, tokens, lemmas in data]
    time_slices = [freq for year, freq in sorted(collections.Counter(data_years).items(), key=lambda x: utils.year_annual_report_comparator(x[0]))]
    assert sum(time_slices) == len(data)

    texts = [sum(x[2], []) for x in data]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=config.MIN_FREQ, no_above=config.MAX_DOC_RATIO)
    corpus = [[dictionary.doc2bow(lemmas) for lemmas in lemmas_sent] for item, tokens, lemmas_sent in data]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return corpus, dictionary, texts, time_slices, data


def get_sentence_document_topic_distribution(num_topics, data, corpus, model, output, save=True):
    if os.path.exists(output):
        return utils.load_pickle(output)

    sentence_documents_topic_distr = []
    for i, (item, tokens_sent, lemmas_sent) in enumerate(data):
        # item = fix_item_path(item, section) # To filter root path

        # Get % of each topic per document
        sent_topic_distr = []
        for j, lemmas in enumerate(lemmas_sent):
            sent_topic_distr.append(model.get_document_topics(corpus[i][j], 1e-10))
        assert len(sent_topic_distr) == len(lemmas_sent)
        assert len(sent_topic_distr) == len(tokens_sent)

        sentence_documents_topic_distr.append((item, sent_topic_distr))

    # Save
    if save:
        with open(output, 'wb') as fp:
            pickle.dump(sentence_documents_topic_distr, fp)

    return sentence_documents_topic_distr


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    master_dictionary, md_header, sentiment_categories, stopwords, total_documents = utils.load_masterdictionary(config.SENTIMENT_LEXICON, True, False, True)

    sections_to_analyze = [config.DATA_7A_FOLDER]
    for section in sections_to_analyze:
        path = os.path.join(config.OUTPUT_FOLDER_SENTIMENT, section[section.rfind('/')+1:])
        if not os.path.exists(path):
            os.makedirs(path)

        data = load_and_clean_data(section)  # Override in order to keep '.' to distinguish sentences
        data, lemma_to_idx, idx_to_lemma = preprocess(section, data)
        corpus, dictionary, texts, time_slices, data = preprocessing_topic(data, idx_to_lemma)

        model_filepath = config.TRAIN_PARAMETERS[section][1]
        assert os.path.exists(model_filepath)
        model, c_v, u_mass = analyze_topics_static.train_topic_model_or_load(corpus, dictionary, texts, model_file=model_filepath, only_viz=True)

        num_topics = config.TRAIN_PARAMETERS[section][0]
        topics = utils.load_pickle(config.TRAIN_PARAMETERS[section][5])
        terms_topics = utils.load_pickle(config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.TERMS_EXTENSION))
        documents_topic_distr = utils.load_pickle(config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.DOCS_EXTENSION))
        sentence_documents_topic_distr = get_sentence_document_topic_distribution(num_topics, data, corpus, model, config.TRAIN_PARAMETERS[section][5].replace(config.TOPIC_EXTENSION, config.SENT_DOCS_EXTENSION))

        # These are NOT pure sentiments
        # Strong modal -> always, definitely, never
        # Moderate model -> can, generally, usually
        # Weak modal -> almost, could, might, suggests
        # Constraining -> commit, encumber, limit
        # Interesting -> provides useful examples of such thins as context or ambiguity

        # Pure sentiments
        # Positive
        # Negative
        # Litigious -> Words reflecting a propsenity for legal contest or, per their label, litigiousness (~731 words)
        #              e.g. claimant, deposition, interlocutory, testimony, tort
        # Uncertainties -> Words denoting uncertainty, with emphasis on the general notion of imprecision rather than exclusively focusing on risk ( ~285 words)
        #                  e.g. approximate, contingency, depend, fluctuate, indefinite, uncertain, variability
        # Use sentiment lexicon to extract polarity
        sentiments = []
        for idx, tokens_sent, lemma_sent in data:
            words_sent = tokens_sent # Might be lemmas

            sentiments_sent = []
            for sent in words_sent:
                sentiments_word = []
                for w in sent:
                    if '_' in w: # Handle n-grams
                        w = w.split('_')
                    else:
                        w = [w]
                    for ww in w:
                        if ww in master_dictionary and not master_dictionary[ww].stopword:
                            sentiments_word.append(master_dictionary[ww])
                sentiments_sent.append(sentiments_word)
            sentiments.append((idx, sentiments_sent))

        # Aggregate
        sentiments_agg = []
        for idx, sentiments_sent in sentiments:
            sents_agg = []
            for sent in sentiments_sent:
                sent_dict = {'negative':0, 'positive':0, 'uncertainty':0, 'litigious':0, 'constraining':0, 'strong_modal':0, 'weak_modal':0, 'neutral':0}
                for w in sent:
                    for k, v in w.sentiment.items():
                        if v:
                            sent_dict[k] += 1
                sents_agg.append(sent_dict)
            sentiments_agg.append(sents_agg)

        for i, (item, tokens_sent, lemma_sent) in enumerate(data):
            file = os.path.join(path, item[item.rfind('/')+1:])
            if '45919_0001193125-11-053421.txt' in file:
                a = 2
            text = data[i][1]
            sents = sentiments[i][1]
            sent_agg = sentiments_agg[i]
            topics = sentence_documents_topic_distr[i][1]
            assert len(text) == len(sents) and len(sents) == len(sent_agg) and len(sent_agg) == len(topics)

            # Find unique sentences
            all_sentences = [' '.join(t) for t in data[i][1]]
            all_sentences_set = set()
            final_indices = set()
            for j, sent in enumerate(all_sentences):
                if sent not in all_sentences_set:
                    all_sentences_set.add(sent)
                    final_indices.add(j)

            with open(file, 'w', encoding='utf-8') as fp:
                fp.write('<Text by sentence>\n')
                for j, sent in enumerate(text):
                    fp.write(str(j) + ') ' + ' '.join(sent) + '\n')
                fp.write('\n')

                fp.write('<Top 5 Topics per sentence>\n')
                for j, ts in enumerate(topics):
                    fp.write(str(j) + ') ')
                    if j in final_indices:
                        fp.write(', '.join([str(t)+':'+str(c) for t,c in sorted(ts, key=lambda x:-float(x[1]))[:5]]) if len(corpus[i][j]) != 0 else 'Empty')
                    else:
                        fp.write('Duplicate')
                    fp.write('\n')
                fp.write('\n')

                fp.write('<Sentiments per sentence>\n')
                for j, sent in enumerate(sents):
                    fp.write(str(j) + ') ')
                    if j in final_indices:
                        fp.write(', '.join([w.word +'(' + ','.join([k for k,v in w.sentiment.items() if v]) + ')' for w in sent]) + '\n')
                    else:
                        fp.write('Duplicate\n')
                fp.write('\n')

                fp.write('<Aggregation sentiments all sentences (with duplicata removed)>\n')
                final_sentiments_agg = {'negative': 0, 'positive': 0, 'uncertainty': 0, 'litigious': 0, 'constraining': 0, 'strong_modal': 0, 'weak_modal': 0, 'neutral': 0}
                for j, sent in enumerate(sent_agg):
                    if j in final_indices:
                        for k, v in sent.items():
                                final_sentiments_agg[k] += v
                fp.write(', '.join([str(k)+':'+str(v) for k,v in sorted(final_sentiments_agg.items(), key=lambda x:-x[1])]) + '\n')
                fp.write('\n')

                fp.write('<Aggregation sentiments all sentences (without duplicata removed)>\n')
                final_sentiments_agg_dup = {'negative': 0, 'positive': 0, 'uncertainty': 0, 'litigious': 0, 'constraining': 0, 'strong_modal': 0, 'weak_modal': 0, 'neutral': 0}
                for j, sent in enumerate(sent_agg):
                    for k, v in sent.items():
                        final_sentiments_agg_dup[k] += v
                fp.write(', '.join([str(k)+':'+str(v) for k,v in sorted(final_sentiments_agg_dup.items(), key=lambda x:-x[1])]) + '\n')
                fp.write('\n')