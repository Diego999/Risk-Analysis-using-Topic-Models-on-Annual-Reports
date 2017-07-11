import os

MULTITHREADING = True

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

# Key words to look up in the headers
KEY_WORDS_LOOKUP_FISCAL_YEAR_END = ['FISCAL YEAR END', 'fiscal year ended', 'fiscal period ended', 'year ended']

KEY_COMPANY_NAME = 'company_name'
KEY_CIK = 'cik'
KEY_FISCAL_YEAR_END = 'fiscal_year_end'
KEY_RELEASED_DATE = 'release_date'

KEY_MAPPING = {k:KEY_FISCAL_YEAR_END for k in KEY_WORDS_LOOKUP_FISCAL_YEAR_END}
KEY_MAPPING_UNIQUE = {v:k for k,v in KEY_MAPPING.items()}
