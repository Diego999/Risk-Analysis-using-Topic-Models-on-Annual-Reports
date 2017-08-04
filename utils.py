import pymysql
import os


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


