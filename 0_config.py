import os
import multiprocessing

SF_NLP_JARS = '/Users/diego/Github/SelfSentRevisited/lib/stanford-corenlp-full-2016-10-31/*'
STOPWORD_LIST = './stopwords_big.txt'
FORCE_PREPROCESSING = False

MULTITHREADING = True
NUM_CORES = multiprocessing.cpu_count()

EXTENSION_10K_REPORT = 'txt'
EXTENSION_CLEAN_PREPROCESSING = 'txt_clean'

DATA_FOLDER = os.path.join('.', 'data/')
DATA_GZ_FOLDER = os.path.join(DATA_FOLDER, 'gz')
DATA_PD_FOLDER = os.path.join(DATA_FOLDER, 'pd')
DATA_AR_FOLDER = os.path.join(DATA_FOLDER, 'ar')
DATA_SECTON_FOLDER = os.path.join(DATA_FOLDER, 'sections')
DATA_1A_FOLDER = os.path.join(DATA_SECTON_FOLDER, '1a_risk_factors')
DATA_7_FOLDER = os.path.join(DATA_SECTON_FOLDER, '7_managements_discussion_and_analysis_of_financial_condition_and_results_of_operations')
DATA_7A_FOLDER = os.path.join(DATA_SECTON_FOLDER, '7a_quantitative_and_qualitative_disclosures_about_market_risk')
SUFFIX_CLEAN_DATA = '_clean.txt'
SUFFIX_PREPROCESSED_DATA = '_preprocessed.pkl'
DICT_LEMMA_IDX = 'dict_lemma_idx.pkl'
DICT_IDX_LEMMA = 'dict_idx_lemma.pkl'

URL_ROOT = 'https://www.sec.gov/Archives/'
URL_INDEX_PATTERN = URL_ROOT + 'edgar/full-index/{year}/QTR{quarter}/company.gz'
PDF_MERGE_FILE = os.path.join(DATA_PD_FOLDER, 'merged_pds.pd')
PDF_MERGE_10K_FILE = os.path.join(DATA_PD_FOLDER, 'merged_10k_pds.pd')
START_YEAR = 1993
START_QUARTER = 1

FORM_TYPE = '10-K'

CIK_COMPANY_NAME_SEPARATOR = '_'
NAME_FILE_PER_CIK = 'names'

LOG_FILE = 'log.txt'
LOG_FISCAL_YEAR_END_MISSING = 'missing_fiscal_year_end.txt'
LOG_RELEASE_DATE_MISSING = 'missing_release_date.txt'

READ_SECTION_EVEN_NO_OFFSET_END = False
ITEM_1A_DIFF_OFFSET_KEEP_BOTH = 500
ITEM_1A_AGG_METHOD = max
ITEM_1A_LINES_TO_READ_IF_NO_NEXT_INDEX = 420 # 75 percentile rank

ITEM_7_DIFF_OFFSET_KEEP_BOTH = -1
ITEM_7_AGG_METHOD = min
ITEM_7_LINES_TO_READ_IF_NO_NEXT_INDEX = 752 # 75 percentile rank

ITEM_7A_DIFF_OFFSET_KEEP_BOTH = -1
ITEM_7A_AGG_METHOD = max
ITEM_7A_LINES_TO_READ_IF_NO_NEXT_INDEX = 49 # 75 percentile rank

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
