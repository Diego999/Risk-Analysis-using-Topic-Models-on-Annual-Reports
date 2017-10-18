import os
import multiprocessing

SF_NLP_JARS = './lib/stanford-corenlp-full-2017-06-09/*'
STOPWORD_LIST = './stopwords_big.txt'

TUNING = False

MULTITHREADING = True
NUM_CORES = multiprocessing.cpu_count()
SEED = 0

DO_NOT_COMPUTE_COHERENCE = False

EXTENSION_10K_REPORT = 'txt'
EXTENSION_CLEAN_PREPROCESSING = 'txt_clean'
TOPIC_EXTENSION = 'topic.pkl'
TERMS_EXTENSION = 'term.pkl'
DOCS_EXTENSION = 'doc.pkl'
SENT_DOCS_EXTENSION = 'sent_doc.pkl'

OUTPUT_FOLDER = os.path.join('.', 'out/')
MODEL_FOLDER = os.path.join('.', 'models/')
DATA_FOLDER = os.path.join('.', 'data/')
DATA_GZ_FOLDER = os.path.join(DATA_FOLDER, 'gz')
DATA_PD_FOLDER = os.path.join(DATA_FOLDER, 'pd')
DATA_AR_FOLDER = os.path.join(DATA_FOLDER, 'ar')
DATA_TEMP_FOLDER = os.path.join(DATA_FOLDER, 'temp')
DATA_SECTON_FOLDER = os.path.join(DATA_FOLDER, 'sections')
DATA_STOCKS_FOLDER = os.path.join(DATA_FOLDER, 'stocks')
DATA_NI_SEQ_FOLDER = os.path.join(DATA_FOLDER, 'ni_seq')
DATA_SEC_IND_FOLDER = os.path.join(DATA_FOLDER, 'sec_ind')
DATA_DISTRIBUTION_FOLDER = os.path.join(OUTPUT_FOLDER, 'distributions')
DATA_TSNE_FOLDER = os.path.join(OUTPUT_FOLDER, 'tsne')
DATA_PCA_FOLDER = os.path.join(OUTPUT_FOLDER, 'pca')
DATA_MDS_FOLDER = os.path.join(OUTPUT_FOLDER, 'mds')
DATA_LTSA_FOLDER = os.path.join(OUTPUT_FOLDER, 'ltsa')
DATA_TSNE_COMPANY_FOLDER = os.path.join(DATA_TSNE_FOLDER, 'company')
DATA_PCA_COMPANY_FOLDER = os.path.join(DATA_PCA_FOLDER, 'company')
DATA_MDS_COMPANY_FOLDER = os.path.join(DATA_MDS_FOLDER, 'company')
DATA_LTSA_COMPANY_FOLDER = os.path.join(DATA_LTSA_FOLDER, 'company')
SENTIMENT_LEXICON = os.path.join(DATA_FOLDER, 'LoughranMcDonald_MasterDictionary_2014.csv')
CIK_TICKER_ADDITIONAL = os.path.join(DATA_FOLDER, 'cik_ticker.csv')
CIK_TICKER_ADDITIONAL_2 = os.path.join(DATA_FOLDER, 'cusip_ticker.csv')
CIK_TICKER_ADDITIONAL_3 = os.path.join(DATA_FOLDER, 'cik_ticker_add.csv')
COMP_SEC_IND_NASDAQ = os.path.join(DATA_FOLDER, 'companylist_nasdaq.csv')
COMP_SEC_IND_AMEX = os.path.join(DATA_FOLDER, 'companylist_amex.csv')
COMP_SEC_IND_NYSE = os.path.join(DATA_FOLDER, 'companylist_nyse.csv')

OUTPUT_FOLDER_SENTIMENT = os.path.join(OUTPUT_FOLDER, 'sentiments')
OUTPUT_FOLDER_WORD_EMB = os.path.join(OUTPUT_FOLDER, 'words_emb_raw_text')

SECTION_1A = '1a_risk_factors'
DATA_1A_FOLDER = os.path.join(DATA_SECTON_FOLDER, SECTION_1A)
DATA_1A_TOPICS = os.path.join(DATA_SECTON_FOLDER, SECTION_1A + '.' + TOPIC_EXTENSION)

SECTION_7 = '7_managements_discussion_and_analysis_of_financial_condition_and_results_of_operations'
DATA_7_FOLDER = os.path.join(DATA_SECTON_FOLDER, SECTION_7)
DATA_7_TOPICS = os.path.join(DATA_SECTON_FOLDER, SECTION_7 + '.' + TOPIC_EXTENSION)

SECTION_7A = '7a_quantitative_and_qualitative_disclosures_about_market_risk'
DATA_7A_FOLDER = os.path.join(DATA_SECTON_FOLDER, SECTION_7A)
DATA_7A_TOPICS = os.path.join(DATA_SECTON_FOLDER, SECTION_7A + '.' + TOPIC_EXTENSION)

SUFFIX_CLEAN_DATA = '_clean.txt'
SUFFIX_PREPROCESSED_DATA = '_preprocessed.pkl'
SUFFIX_PREPROCESSED_DATA_FOR_SENT = '_for_sent.pkl'
SUFFIX_PREPROCESSED_DATA_FOR_WE = '_for_we.pkl'
SUFFIX_INPUT_DATA = '_input'
SUFFIX_DF = '_df'
DICT_LEMMA_IDX = '_dict_lemma_idx.pkl'
DICT_IDX_LEMMA = '_dict_idx_lemma.pkl'

MIN_FREQ = 20
MIN_FREQ_BI = 20
MIN_FREQ_TRI = 10
MIN_FREQ_QUA = 5
MAX_DOC_RATIO = 0.5

URL_ROOT = 'https://www.sec.gov/Archives/'
URL_INDEX_PATTERN = URL_ROOT + 'edgar/full-index/{year}/QTR{quarter}/company.gz'
PDF_MERGE_FILE = os.path.join(DATA_PD_FOLDER, 'merged_pds.pd')
PDF_MERGE_10K_FILE = os.path.join(DATA_PD_FOLDER, 'merged_10k_pds.pd')
START_YEAR = 1993
START_YEAR_TWO_DIGIT = int(str(START_YEAR)[-2:])
START_QUARTER = 1

FORM_TYPE = '10-K'

CIK_COMPANY_NAME_SEPARATOR = '_'
NAME_FILE_PER_CIK = 'names'

LOG_FILE = 'log.txt'
LOG_FISCAL_YEAR_END_MISSING = 'missing_fiscal_year_end.txt'
LOG_RELEASE_DATE_MISSING = 'missing_release_date.txt'

READ_SECTION_EVEN_NO_OFFSET_END = False


ITEM_1A_TOPICS = 66 # to be enough fine-grained. o/w 22
ITEM_1A_PARAMS = None
if ITEM_1A_TOPICS == 22:
    ITEM_1A_PARAMS = {'num_topics':ITEM_1A_TOPICS, 'alpha':'asymmetric', 'eta':'auto', 'chunksize':2000, 'decay':0.6, 'offset':2.0, 'passes':10, 'iterations':400, 'eval_every':10}
elif ITEM_1A_TOPICS == 66:
    ITEM_1A_PARAMS = {'num_topics':ITEM_1A_TOPICS, 'alpha':'symmetric', 'eta':'symmetric', 'chunksize':2000, 'decay':0.6, 'offset':8.0, 'passes':10, 'iterations':400, 'eval_every':10}

ITEM_1A_MODEL = os.path.join(MODEL_FOLDER, 'lda_1a_{}.model'.format(ITEM_1A_TOPICS)) if ITEM_1A_PARAMS is None else os.path.join(MODEL_FOLDER, 'lda_1a_{}_best.model'.format(ITEM_1A_TOPICS))
ITEM_1A_MODEL_VIZ = ITEM_1A_MODEL.replace('.model', '.html')
ITEM_1A_MODEL_DYN = ITEM_1A_MODEL.replace('.model', '.model_dyn')
ITEM_1A_DIFF_OFFSET_KEEP_BOTH = 500
ITEM_1A_AGG_METHOD = max
ITEM_1A_LINES_TO_READ_IF_NO_NEXT_INDEX = 420 # 75 percentile rank
KEYWORDS_TO_DETECT_1A = ['smaller reporting company', 'pages', 'discussion and analysis of financial condition']
MIN_LENGTH_EMPTY_1A = 400 # MIN chars in the section to be kept
MIN_LENGTH_KEYWORDS_1A = 5000 # For all texts having less than MIN_LENGTH_KEYWORDS chars and less than MIN_LINES_KEYWORDS lines, remove them if they contain KEYWORDS_TO_DETECT_1A
MIN_LINES_KEYWORDS_1A = 50
REMOVE_START_WORDS_1A = ['item #a risk factors risk factors',
                         'item #a risk factors',
                         'item #arisk factors',
                         'item #a - risk factors',
                         'risk factors and uncertainties',
                         'risk factors risk factors',
                         'risk factors',
                         'isk factors and uncertainties',
                         'isk factors risk factors',
                         'isk factors',
                         'item #a',
                         'item #']

ITEM_7_TOPICS = 54
ITEM_7_PARAMS = None
if ITEM_7_TOPICS == 54:
    ITEM_7_PARAMS = {'num_topics':ITEM_7_TOPICS, 'alpha':'asymmetric', 'eta':'auto', 'chunksize':2000, 'decay':0.6, 'offset':16.0, 'passes':10, 'iterations':400, 'eval_every':10}

ITEM_7_MODEL = os.path.join(MODEL_FOLDER, 'lda_7_{}.model'.format(ITEM_7_TOPICS)) if ITEM_7_PARAMS is None else os.path.join(MODEL_FOLDER, 'lda_7_{}_best.model'.format(ITEM_7_TOPICS))
ITEM_7_MODEL_VIZ = ITEM_7_MODEL.replace('.model', '.html')
ITEM_7_MODEL_DYN = ITEM_7_MODEL.replace('.model', '.model_dyn')
ITEM_7_DIFF_OFFSET_KEEP_BOTH = -1
ITEM_7_AGG_METHOD = min
ITEM_7_LINES_TO_READ_IF_NO_NEXT_INDEX = 752 # 75 percentile rank
KEYWORDS_TO_DETECT_7 = ['pages', 'herein by reference']
MIN_LENGTH_EMPTY_7 = 1000 # MIN chars in the section to be kept
MIN_LENGTH_KEYWORDS_7 = 5000 # For all texts having less than MIN_LENGTH_KEYWORDS chars and less than MIN_LINES_KEYWORDS lines, remove them if they contain KEYWORDS_TO_DETECT_1A
MIN_LINES_KEYWORDS_7 = 50
REMOVE_START_WORDS_7 = sorted(['item # management s discussion and analysis of financial condition and results of operation',
                        'item # management s discussion analysis of financial conditions results of operation',
                        'item # management s discussion and analysis of financial condition and results of operation',
                        'item # management s discussion and analysis md a of financial condition and results of operation',
                        'item # - management s discussion and analysis of financial condition and results of operation',
                        'item # management s discussion and analysis md a of financial condition and results of operation',
                        'item # management s discussion and analysis of consolidated financial condition and results of operation',
                        'item # management s discussion and analysis of financial condition and results of operations',
                        'item # management s discussion analysis of financial conditions results of operations',
                        'item # management s discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis md a of financial condition and results of operations',
                        'item # - management s discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis md a of financial condition and results of operations',
                        'item # management s discussion and analysis of consolidated financial condition and results of operations',
                        "item # management's discussion and analysis of financial condition and results of operation",
                        "item # management's discussion analysis of financial conditions results of operation",
                        "item # management's discussion and analysis of financial condition and results of operation",
                        "item # management's discussion and analysis md a of financial condition and results of operation",
                        "item # - management's discussion and analysis of financial condition and results of operation",
                        "item # management's discussion and analysis md a of financial condition and results of operation",
                        "item # management's discussion and analysis of consolidated financial condition and results of operation",
                        "item # management's discussion and analysis of financial condition and results of operations",
                        "item # management's discussion analysis of financial conditions results of operations",
                        "item # management's discussion and analysis of financial condition and results of operations",
                        "item # management's discussion and analysis md a of financial condition and results of operations",
                        "item # - management's discussion and analysis of financial condition and results of operations",
                        "item # management's discussion and analysis md a of financial condition and results of operations",
                        "item # management's discussion and analysis of consolidated financial condition and results of operations",
                        "management's discussion and analysis of financial condition - and results of operations",
                        "management's discussion and analysis of financial condition - and results of operation",
                        'management s discussion and analysis of financial condition - and results of operations',
                        'managemen s discussion and analysis of financial condition - and results of operation',
                        "management's discussion and analysis of financial condition and - - results of operations - -",
                        'management s discussion and analysis of results of operations and financial condition',
                        'management s discussion and analysis of financial condition and operating results',
                        'management s discussion and analysis of financial condition and results of operation',
                        'management s discussion analysis of financial conditions results of operation',
                        'management s discussion and analysis of financial condition and results of operation',
                        'management s discussion and analysis md a of financial condition and results of operation',
                        '- management s discussion and analysis of financial condition and results of operation',
                        'management s discussion and analysis md a of financial condition and results of operation',
                        'management s discussion and analysis of consolidated financial condition and results of operation',
                        'management s discussion and analysis of financial condition and results of operations',
                        'management s discussion analysis of financial conditions results of operations',
                        'management s discussion and analysis of financial condition and results of operations',
                        'management s discussion and analysis md a of financial condition and results of operations',
                        '- management s discussion and analysis of financial condition and results of operations',
                        'management s discussion and analysis md a of financial condition and results of operations',
                        'management s discussion and analysis of consolidated financial condition and results of operations',
                        "management's discussion and analysis of financial condition and results of operation",
                        "management's discussion analysis of financial conditions results of operation",
                        "management's discussion and analysis of financial condition and results of operation",
                        "management's discussion and analysis md a of financial condition and results of operation",
                        "- management's discussion and analysis of financial condition and results of operation",
                        "management's discussion and analysis md a of financial condition and results of operation",
                        "management's discussion and analysis of consolidated financial condition and results of operation",
                        "management's discussion and analysis of financial condition and results of operations",
                        "management's discussion analysis of financial conditions results of operations",
                        "management's discussion and analysis of financial condition and results of operations",
                        "management's discussion and analysis md a of financial condition and results of operations",
                        "- management's discussion and analysis of financial condition and results of operations",
                        "management's discussion and analysis md a of financial condition and results of operations",
                        "management's discussion and analysis of consolidated financial condition and results of operations",
                        'management discussion and analysis of financial condition and results of operations',
                        "management's discussion and analysis of financial condition and - - results of operations",
                        "item # management's discussion and analysis of financial condition and results of operations",
                        'item # management s discussion and analysis of financial condition and results of operation',
                        "item # management's discussion and analysis of financial condition and results of operations",
                        'item # management s discussion and analysis of financial condition and results of operation',
                        "management's discussion and analysis of financial conduction and results of operations",
                        "management's discussion and analysis of financial conduction and results of operation",
                        'management s discussion and analysis of financial conduction and results of operations',
                        'management s discussion and analysis of financial conduction and results of operation',
                        'management discussion and analysis of financial condition and results of operations',
                        "management's discussion and analysis of financial condition and results of operations",
                        'management s discussion and analysis of financial condition and results of operations',
                        "management's discussion and analysis of financial condition and - - - results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial condition and results - of operations",
                        "management's discussion and analysis of financial condition and results - of operations",
                        "management's discussion and analysis of financial condition and results - - of operations",
                        "management's discussion and analysis of financial condition and results - of operations",
                        "management's discussion and analysis of financial condition and result of operations",
                        "management's discussion and analysis of financial - - condition and results of operations",
                        'management discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis of financial condition and results of operations',
                        'management s discussion and analysis of results of operations and financial condition',
                        "management's discussion and analysis of financial conduction and results of operations",
                        "management's discussion and analysis of financial condition and result of operations",
                        "management' discussion and analysis of financial condition and results of operations",
                        "management's discussion and analysis of financial condition and - - - results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial - condition and results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial condition and - results of operations",
                        "management's discussion and analysis of financial condition and results - of operations",
                        "management's discussion and analysis of financial condition and results - of operations",
                        'item # management discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis of results of operations and financial condition',
                        'item # management discussion and analysis of financial condition and results of operations',
                        'item # management s discussion and analysis of financial condition and results of operations',
                        "item # - management's discussion and analysis of financial conduction and results of operations",
                        "item # management's discussion and analysis of financial condition and result of operations",
                        "item #- management' discussion and analysis of financial condition and results of operations",
                        "item # management's discussion and analysis of financial condition and - - - results of operations",
                        "item # management s discussion and analysis or plan of operations plan of operation",
                        "item # management's discussion and analysis of financial condition and - results of operations",
                        "item # management's discussion and analysis of financial condition and results - of operations",
                        "item # management's discussion and analysis of financial condition and - results of operations",
                        "item # management's discussion and analysis of financial condition and results - of operations",
                        "item # management's discussion and analysis of financial condition and result of operations",
                        "management discussion and analysis of financial condition and results of operation",
                        "management's discussion and analysis of financial condition and results - - of operations",
                        "management's discussion and analysis of financial conditions and results of operations",
                        "management's discussion and analysis of financial condition and plan of operation",
                        "management's discussion and analysis of financial condition and - - results of operations",
                        "management's discussion and analysis of financial - - condition and results of operations",
                        "management's discussion and analysis of financial condition and results - - of operations",
                        "management's discussion and analysis of financial condition and results - - of operations",
                        "management's discussion and analysis of the results of operations",
                        "item # management s discussion and analysis of financial condition and results of operations",
                        "management's discussion and analysis of financial condition and - - results of operations",
                        'item #'], key=lambda x:-len(x))

ITEM_7A_TOPICS = 22 # or 19/22
ITEM_7A_PARAMS = None
if ITEM_7A_TOPICS == 19:
    ITEM_7A_PARAMS = {'num_topics':ITEM_7A_TOPICS, 'alpha':'symmetric', 'eta':'auto', 'chunksize':2000, 'decay':0.6, 'offset':64.0, 'passes':10, 'iterations':400, 'eval_every':10}
elif ITEM_7A_TOPICS == 22:
    ITEM_7A_PARAMS = {'num_topics':ITEM_7A_TOPICS, 'alpha':'symmetric', 'eta':'symmetric', 'chunksize':2000, 'decay':0.5, 'offset':8.0, 'passes':10, 'iterations':400, 'eval_every':10}

ITEM_7A_MODEL = os.path.join(MODEL_FOLDER, 'lda_7a_{}.model'.format(ITEM_7A_TOPICS)) if ITEM_7A_PARAMS is None else os.path.join(MODEL_FOLDER, 'lda_7a_{}_best.model'.format(ITEM_7A_TOPICS))
ITEM_7A_MODEL_VIZ = ITEM_7A_MODEL.replace('.model', '.html')
ITEM_7A_MODEL_DYN = ITEM_7A_MODEL.replace('.model', '.model_dyn')
ITEM_7A_DIFF_OFFSET_KEEP_BOTH = -1
ITEM_7A_AGG_METHOD = max
ITEM_7A_LINES_TO_READ_IF_NO_NEXT_INDEX = 49 # 75 percentile rank
KEYWORDS_TO_DETECT_7A = ['disclose the applicable market risk', 'pages', 'discussion and analysis of financial condition', 'discussion and analysis of results', 'herein by reference']
MIN_LENGTH_EMPTY_7A = 411 # MIN chars in the section to be kept
MIN_LENGTH_KEYWORDS_7A = 1000 # For all texts having less than MIN_LENGTH_KEYWORDS chars and less than MIN_LINES_KEYWORDS lines, remove them if they contain KEYWORDS_TO_DETECT_1A
MIN_LINES_KEYWORDS_7A = 10
REMOVE_START_WORDS_7A = ['item #a - quantitative and qualitative disclosures about market risk -',
                         'item #a quantitative and qualitative disclosures about market risk',
                         'item #a quantitative and qualitative disclosure about market risk',
                         'item #a qualitative and quantitative disclosures about market risk',
                         'item #a qualitative and quantitative disclosure about market risk',
                         'item #a quantative and qualitive disclosures about market risk',
                         'item #a quantitative and qualitative disclosures about market',
                         'item #a quantitative and qualitative disclosure about market',
                         'item #a qualitative and quantitative disclosures about market',
                         'item #a qualitative and quantitative disclosure about market',
                         'item #a - quantitative and qualitative disclosures about market risk',
                         'quantitative and qualitative disclosures about market risk -',
                         'quantitative and qualitative disclosures about market risk',
                         'quantitative and qualitative disclosure about market risk',
                         'qualitative and quantitative disclosures about market risk',
                         'qualitative and quantitative disclosure about market risk',
                         'quantative and qualitive disclosures about market risk',
                         'quantitative and qualitative disclosures about market',
                         'quantitative and qualitative disclosure about market',
                         'qualitative and quantitative disclosures about market',
                         'qualitative and quantitative disclosure about market',
                         'quantitative and qualitative disclosures about market risk',
                         'uantitative and qualitative disclosures about market risk -',
                         'uantitative and qualitative disclosures about market risk',
                         'uantitative and qualitative disclosure about market risk',
                         'ualitative and quantitative disclosures about market risk',
                         'ualitative and quantitative disclosure about market risk',
                         'uantative and qualitive disclosures about market risk',
                         'uantitative and qualitative disclosures about market',
                         'uantitative and qualitative disclosure about market',
                         'ualitative and quantitative disclosures about market',
                         'ualitative and quantitative disclosure about market',
                         'uantitative and qualitative disclosures about market risk',
                         'item #a market risks',
                         'item #a market',
                         'item #a',
                         'item #'
                         ]

CLEAN_PARAMETERS = {DATA_1A_FOLDER: [KEYWORDS_TO_DETECT_1A, MIN_LENGTH_EMPTY_1A, MIN_LENGTH_KEYWORDS_1A, MIN_LINES_KEYWORDS_1A, REMOVE_START_WORDS_1A],
                    DATA_7_FOLDER: [KEYWORDS_TO_DETECT_7, MIN_LENGTH_EMPTY_7, MIN_LENGTH_KEYWORDS_7, MIN_LINES_KEYWORDS_7, REMOVE_START_WORDS_7],
                    DATA_7A_FOLDER: [KEYWORDS_TO_DETECT_7A, MIN_LENGTH_EMPTY_7A, MIN_LENGTH_KEYWORDS_7A, MIN_LINES_KEYWORDS_7A, REMOVE_START_WORDS_7A]}

TRAIN_PARAMETERS = {DATA_1A_FOLDER: [ITEM_1A_TOPICS, ITEM_1A_MODEL, ITEM_1A_MODEL_VIZ, ITEM_1A_MODEL_DYN, ITEM_1A_PARAMS, os.path.join(DATA_DISTRIBUTION_FOLDER, SECTION_1A + '_{}.{}'.format(ITEM_1A_TOPICS, TOPIC_EXTENSION))],
                    DATA_7_FOLDER: [ITEM_7_TOPICS, ITEM_7_MODEL, ITEM_7_MODEL_VIZ, ITEM_7_MODEL_DYN, ITEM_7_PARAMS, os.path.join(DATA_DISTRIBUTION_FOLDER, SECTION_7 + '_{}.{}'.format(ITEM_7_TOPICS, TOPIC_EXTENSION))],
                    DATA_7A_FOLDER: [ITEM_7A_TOPICS, ITEM_7A_MODEL, ITEM_7A_MODEL_VIZ, ITEM_7A_MODEL_DYN, ITEM_7A_PARAMS, os.path.join(DATA_DISTRIBUTION_FOLDER, SECTION_7A + '_{}.{}'.format(ITEM_7A_TOPICS, TOPIC_EXTENSION))]}

# Key words to look up
KEY_WORDS_LOOKUP_FISCAL_YEAR_END = ['FISCAL YEAR END', 'CONFORMED PERIOD OF REPORT', 'fiscal year ended', 'fiscal period ended', 'year ended']
KEY_WORDS_LOOKUP_RELEASE_DATE_END = ['FILED AS OF DATE', 'Outstanding at', 'DATE', 'thereunto duly authorized', 'undersigned, thereunto duly', 'in the capacities indicated on the', 'thereto duly authorized', 'has been signed below on behalf', 'unto duly authorized', 'consent of independent auditors', 'duly authorized', 'pursuant to the requirements of', 'aggregate market value of the common stock']
KEY_WORDS_ITEM_1A = ['item 1a -', 'item 1a:', 'item 1a.', 'item 1a', 'item no. 1a', 'risk factors']
KEY_WORDS_ITEM_1B = ['item 1b -', 'item 1b:', 'item 1b.', 'item 1b', 'item no. 1b', 'unresolved staff comments'] # To know where to stop for 1A
KEY_WORDS_ITEM_2 = ['item 2 -', 'item 2:', 'item 2.', 'item 2', 'item two', 'item no. 2', 'properties', 'DESCRIPTION OF PROPERTY'.lower()]# To know where to stop for 1A if 1B does not exists
KEY_WORDS_ITEM_7 = ['item 7 -', 'item 7:', 'item 7.', 'item 7', 'item seven', 'item no. 7', 'analysis of financial condition', 'discussion and analysis of financial', 'discussion and analysis of annual report', 'financial condition and results of', 'discussion and analysis of financial condition and results of operations']
KEY_WORDS_ITEM_7A = ['item 7a -', 'item 7a:', 'item 7a.', 'item 7a', 'item no. 7a', 'quantitative and qualitative disclosures about market risk']
KEY_WORDS_ITEM_8 = ['item 8 -', 'item 8:', 'item 8.', 'item 8', 'item eight', 'item no. 8', 'financial statements and supplementary data'] # To know where to stop for 7 and 7A (this section might be also interesting)
KEY_WORDS_ITEM_9 = ['item 9 -', 'item 9:', 'item 9.', 'item 9', 'item nine', 'item no. 9', 'CHANGES IN AND DISAGREEMENTS WITH'.lower(), 'CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND'.lower()] # To know where to stop for
KEY_WORDS_MAX_LENGTH = [max([len(kw) for kw in kws])for kws in [KEY_WORDS_ITEM_1A, KEY_WORDS_ITEM_1B, KEY_WORDS_ITEM_2, KEY_WORDS_ITEM_7, KEY_WORDS_ITEM_7A, KEY_WORDS_ITEM_8, KEY_WORDS_ITEM_9]]
KEY_FILE = 'file'
KEY_COMPANY_NAME = 'company_names'
KEY_CIK = 'cik'
KEY_FISCAL_YEAR_END = 'fiscal_year_end'
KEY_RELEASED_DATE = 'release_date'
KEY_ITEM_1A_1,  KEY_ITEM_1A_2, KEY_ITEM_1A_3 = 'item_1a_start_1', 'item_1a_start_2', 'item_1a_start_3'
KEY_ITEM_1B_1,  KEY_ITEM_1B_2, KEY_ITEM_1B_3 = 'item_1b_start_1', 'item_1b_start_2', 'item_1b_start_3'
KEY_ITEM_2_1,  KEY_ITEM_2_2, KEY_ITEM_2_3 = 'item_2_start_1', 'item_2_start_2', 'item_2_start_3'
KEY_ITEM_7_1,  KEY_ITEM_7_2, KEY_ITEM_7_3 = 'item_7_start_1', 'item_7_start_2', 'item_7_start_3'
KEY_ITEM_7A_1,  KEY_ITEM_7A_2, KEY_ITEM_7A_3 = 'item_7a_start_1', 'item_7a_start_2', 'item_7a_start_3'
KEY_ITEM_8_1,  KEY_ITEM_8_2, KEY_ITEM_8_3 = 'item_8_start_1', 'item_8_start_2', 'item_8_start_3'
KEY_ITEM_9_1,  KEY_ITEM_9_2, KEY_ITEM_9_3 = 'item_9_start_1', 'item_9_start_2', 'item_9_start_3'
KEY_BADLY_PARSED = 'badly_parsed'

KEY_MAPPING = {k:KEY_FISCAL_YEAR_END for k in KEY_WORDS_LOOKUP_FISCAL_YEAR_END}
KEY_MAPPING_UNIQUE = {v:k for k,v in KEY_MAPPING.items()}

# Parsing options
MAX_EMPTY_LINES = 3
MIN_LENGTH_LINE = 4
