import pymysql
import os
import pickle
config = __import__('0_config')


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


