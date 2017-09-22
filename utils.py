import pymysql
import os
import pickle
import random
config = __import__('0_config')


# Simple uniform sampling between[0, len(array)-1]
def draw_val(array):
    return array[random.randint(0, len(array)-1)]


def extract_year_from_filename_annual_report(filename):
    return int(filename.split('-')[-2])


def year_annual_report_comparator(year):
    return year + (2000 if year < config.START_YEAR_TWO_DIGIT else 1900)


def save_pickle(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


# Split list every 10'000 elements if contains more than 30'000 elements
def save_pickle_big_list(data_in, filename):
    final_data = [data_in]
    if len(data_in) > 10000:
        final_data = chunks(data_in, 1000)

    for i, data in enumerate(final_data):
        filename_i = filename + '_' + str(i)
        with open(filename_i, 'wb') as fp:
            pickle.dump(data, fp)


def load_pickle_big_list(filename_in):
    data = []

    filenames = []
    if not os.path.exists(filename_in):
        i = 0
        while os.path.exists(filename_in + '_' + str(i)):
            filenames.append(filename_in + '_' + str(i))
            i += 1
    else:
        filenames = [filename_in]

    for filename in filenames:
        with open(filename, 'rb') as fp:
            data += pickle.load(fp)

    return data


# Yield successive n-sized chunks from l.
def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        res.append(l[i:i + n])
    return res


def create_mysql_connection(all_in_mem=True):
    return pymysql.connect(host='localhost',
                             user=os.getenv('MYSQL_USER'),
                             password=os.getenv('MYSQL_PASSWORD'),
                             db='SEC',
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor if all_in_mem else pymysql.cursors.SSDictCursor)


# Initially from http://sraf.nd.edu/textual-analysis/code/
# Code has been modified


def load_masterdictionary(file_path, print_flag=False, f_log=None, get_other=False):
    _master_dictionary = {}
    _sentiment_categories = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining',
                             'strong_modal', 'weak_modal']
    # Load slightly modified nltk stopwords.  I do not use nltk import to avoid versioning errors.
    # Dropped from nltk: A, I, S, T, DON, WILL, AGAINST
    # Added: AMONG,
    _stopwords = [sw.lower() for sw in ['ME', 'MY', 'MYSELF', 'WE', 'OUR', 'OURS', 'OURSELVES', 'YOU', 'YOUR', 'YOURS',
                       'YOURSELF', 'YOURSELVES', 'HE', 'HIM', 'HIS', 'HIMSELF', 'SHE', 'HER', 'HERS', 'HERSELF',
                       'IT', 'ITS', 'ITSELF', 'THEY', 'THEM', 'THEIR', 'THEIRS', 'THEMSELVES', 'WHAT', 'WHICH',
                       'WHO', 'WHOM', 'THIS', 'THAT', 'THESE', 'THOSE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'BE',
                       'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'HAVING', 'DO', 'DOES', 'DID', 'DOING', 'AN',
                       'THE', 'AND', 'BUT', 'IF', 'OR', 'BECAUSE', 'AS', 'UNTIL', 'WHILE', 'OF', 'AT', 'BY',
                       'FOR', 'WITH', 'ABOUT', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE',
                       'AFTER', 'ABOVE', 'BELOW', 'TO', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER',
                       'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY',
                       'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH',
                       'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN',
                       'JUST', 'SHOULD', 'NOW']]

    with open(file_path) as f:
        _total_documents = 0
        _md_header = f.readline()
        for line in f:
            cols = line.split(',')
            key = cols[0].lower()
            _master_dictionary[key] = MasterDictionary(cols, _stopwords)
            _total_documents += _master_dictionary[key].doc_count
            if len(_master_dictionary) % 5000 == 0 and print_flag:
                print('\r ...Loading Master Dictionary' + ' {}'.format(len(_master_dictionary)), end='', flush=True)

    if print_flag:
        print('\r', end='')  # clear line
        print('\nMaster Dictionary loaded from file: \n  ' + file_path)
        print('  {0:,} words loaded in master_dictionary.'.format(len(_master_dictionary)) + '\n')

    if f_log:
        try:
            f_log.write('\n\n  load_masterdictionary log:')
            f_log.write('\n    Master Dictionary loaded from file: \n       ' + file_path)
            f_log.write('\n    {0:,} words loaded in master_dictionary.\n'.format(len(_master_dictionary)))
        except Exception as e:
            print('Log file in load_masterdictionary is not available for writing')
            print('Error = {0}'.format(e))

    if get_other:
        return _master_dictionary, _md_header.strip().split(','), _sentiment_categories, _stopwords, _total_documents
    else:
        return _master_dictionary


def create_sentimentdictionaries(_master_dictionary, _sentiment_categories):

    _sentiment_dictionary = {}
    for category in _sentiment_categories:
        _sentiment_dictionary[category] = {}
    # Create dictionary of sentiment dictionaries with count set = 0
    for word in _master_dictionary.keys():
        for category in _sentiment_categories:
            if _master_dictionary[word].sentiment[category]:
                _sentiment_dictionary[category][word] = 0

    return _sentiment_dictionary


class MasterDictionary:
    def __init__(self, cols, _stopwords):
        self.word = cols[0].lower()
        self.sequence_number = int(cols[1])
        self.word_count = int(cols[2])
        self.word_proportion = float(cols[3])
        self.average_proportion = float(cols[4])
        self.std_dev_prop = float(cols[5])
        self.doc_count = int(cols[6])
        self.negative = int(cols[7])
        self.positive = int(cols[8])
        self.uncertainty = int(cols[9])
        self.litigious = int(cols[10])
        self.constraining = int(cols[11])
        self.superfluous = int(cols[12])
        self.interesting = int(cols[13])
        self.modal_number = int(cols[14])
        self.strong_modal = False
        if int(cols[14]) == 1:
            self.strong_modal = True
        self.moderate_modal = False
        if int(cols[14]) == 2:
            self.moderate_modal = True
        self.weak_modal = False
        if int(cols[14]) == 3:
            self.weak_modal = True
        self.sentiment = {}
        self.sentiment['negative'] = bool(self.negative)
        self.sentiment['positive'] = bool(self.positive)
        self.sentiment['uncertainty'] = bool(self.uncertainty)
        self.sentiment['litigious'] = bool(self.litigious)
        self.sentiment['constraining'] = bool(self.constraining)
        self.sentiment['strong_modal'] = bool(self.strong_modal)
        self.sentiment['weak_modal'] = bool(self.weak_modal)
        self.sentiment['neutral'] = sum([1 if x else 0 for x in list(self.sentiment.values())]) == 0
        self.neutral = self.sentiment['neutral']
        self.irregular_verb = int(cols[15])
        self.harvard_iv = int(cols[16])
        self.syllables = int(cols[17])
        self.source = cols[18]

        if self.word in _stopwords:
            self.stopword = True
        else:
            self.stopword = False
        return

