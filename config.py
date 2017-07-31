import os
import multiprocessing

MULTITHREADING = True
NUM_CORES = multiprocessing.cpu_count()

DATA_FOLDER = os.path.join('.', 'data/')
DATA_GZ_FOLDER = os.path.join(DATA_FOLDER, 'gz')
DATA_PD_FOLDER = os.path.join(DATA_FOLDER, 'pd')
DATA_AR_FOLDER = os.path.join(DATA_FOLDER, 'ar')
DATA_COMPANY_FOLDER = os.path.join(DATA_FOLDER, 'company')

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
KEY_COMPANY_NAME = 'company_name'
KEY_CIK = 'cik'
KEY_FISCAL_YEAR_END = 'fiscal_year_end'
KEY_RELEASED_DATE = 'release_date'

KEY_MAPPING = {k:KEY_FISCAL_YEAR_END for k in KEY_WORDS_LOOKUP_FISCAL_YEAR_END}
KEY_MAPPING_UNIQUE = {v:k for k,v in KEY_MAPPING.items()}

# Parsing options
MAX_EMPTY_LINES = 3
MIN_LENGTH_LINE = 4