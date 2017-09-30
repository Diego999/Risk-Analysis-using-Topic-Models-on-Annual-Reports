import wrds
import datetime
import math
import utils
import requests
from bs4 import BeautifulSoup
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')

EXTRA_KEYS = ['cusip', 'gvkey', 'lpermno', 'lpermco', 'tic']


def get_cik_lookup(db, connection):
    already_computed = False

    # Check if database already contains the mapping
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as cnt from `companies`;"
        cursor.execute(sql)
        already_computed = cursor.fetchone()['cnt'] != 0

    cik_2_oindices = {}
    if not already_computed:
        # Look up indices from WRDS
        lookup_table = db.raw_sql('select ' + ', '.join(['cik'] + EXTRA_KEYS) + ' from crsp.ccm_lookup where cik is not null')
        for index, row in lookup_table.iterrows():
            cik = row['cik'].rjust(10, '0')
            oindices_utils = {k: str(row[k]) for k in EXTRA_KEYS}
            oindices = {k:(set() if v.lower() == 'nan' else set([v])) for k,v in oindices_utils.items()}

            # Create the hashtable
            if cik not in cik_2_oindices:
                cik_2_oindices[cik] = oindices
            else:
                for k, v in oindices.items():
                    if k not in cik_2_oindices[cik]:
                        cik_2_oindices[cik][k] = v
                    else:
                        cik_2_oindices[cik][k] = cik_2_oindices[cik][k].union(oindices[k])
        # Insert into DB
        with connection.cursor() as cursor:
            for cik, keys in cik_2_oindices.items():
                sql = "INSERT INTO `companies` ({}) VALUES ({})"
                sql_fina_keys = ['cik']
                sql_final_values = [cik]
                for k, v in keys.items():
                    sql_fina_keys += [k]
                    sql_final_values += [', '.join(v)]
                sql = sql.format(', '.join(sql_fina_keys), ', '.join(['%s'] * len(sql_fina_keys)))
                cursor.execute(sql, tuple(sql_final_values))
        connection.commit()

        # Also add extra informations
        extra_cik_2_tic = {}
        with open(config.CIK_TICKER_ADDITIONAL, 'r') as fp:
            for l in fp:
                cik, tic = l.strip().split('|')
                cik = cik.rjust(10, '0')
                assert cik not in extra_cik_2_tic
                extra_cik_2_tic[cik] = {'tic':tic}
        # Insert into DB
        with connection.cursor() as cursor:
            for cik, v in extra_cik_2_tic.items():
                sql = 'SELECT cik, tic FROM companies WHERE cik = %s;'
                cursor.execute(sql, cik)
                rows = cursor.fetchall()
                if len(rows) == 0:
                    sql = "INSERT INTO `companies` (`cik`, `tic`) VALUES (%s, %s);"
                    cursor.execute(sql, (cik, v['tic']))
        connection.commit()

        # Also add extra informations 2
        extra_cik_2_tic_2 = {}
        with open(config.CIK_TICKER_ADDITIONAL_2, 'r') as fp:
            for l in fp:
                tic, cusip, cik = l.strip().split('|')[1:]
                if cik == '':
                    continue
                cik = cik.rjust(10, '0')
                if cik not in extra_cik_2_tic_2:
                    extra_cik_2_tic_2[cik] = {'tic': set([tic]), 'cusip': set([cusip])}
                else:
                    extra_cik_2_tic_2[cik]['tic'] = extra_cik_2_tic_2[cik]['tic'].union(set([tic]))
                    extra_cik_2_tic_2[cik]['cusip'] = extra_cik_2_tic_2[cik]['cusip'].union(set([cusip]))
        # Insert into DB
        with connection.cursor() as cursor:
            for cik, v in extra_cik_2_tic_2.items():
                sql = 'SELECT cik FROM companies WHERE cik = %s;'
                cursor.execute(sql, cik)
                rows = cursor.fetchall()
                if len(rows) == 0:
                    sql = "INSERT INTO `companies` (`cik`, `tic`, `cusip`) VALUES (%s, %s, %s);"
                    cursor.execute(sql, (cik, ', '.join(v['tic']), [', '.join(v['cusip'])]))
        connection.commit()

        # Also add extra informations 3
        extra_cik_2_tic_3 = {}
        with open(config.CIK_TICKER_ADDITIONAL_3, 'r') as fp:
            for l in fp:
                try:
                    cik, tic = l.strip().split('|')
                except:
                    a = 2
                cik = str(cik).rjust(10, '0')
                if cik in extra_cik_2_tic_3:
                    extra_cik_2_tic_3[cik]['tic'] = extra_cik_2_tic_3[cik]['tic'].union(set([tic]))
                else:
                    extra_cik_2_tic_3[cik] = {'tic': set(tic)}
        # Insert into DB
        with connection.cursor() as cursor:
            for cik, v in extra_cik_2_tic_3.items():
                sql = 'SELECT cik, tic FROM companies WHERE cik = %s;'
                cursor.execute(sql, cik)
                rows = cursor.fetchall()
                if len(rows) == 0:
                    sql = "INSERT INTO `companies` (`cik`, `tic`) VALUES (%s, %s);"
                    cursor.execute(sql, (cik, ', '.join(v['tic'])))
        connection.commit()

        return {**cik_2_oindices, **extra_cik_2_tic, **extra_cik_2_tic_2, **extra_cik_2_tic_3}, already_computed
    else:
        with connection.cursor() as cursor:
            sql = "SELECT distinct cik, tic, cusip, gvkey, lpermno, lpermco FROM companies;"
            cursor.execute(sql)
            for row in cursor.fetchall():
                cik = row['cik']
                assert cik not in cik_2_oindices
                cik_2_oindices[cik] = {}
                for k, v in row.items():
                    if k not in ['cik'] and v != '' and v is not None:
                        cik_2_oindices[cik][k] = (set([vv.strip() for vv in v.split(',')]) if ',' in v else set([v]))
        return cik_2_oindices, already_computed


# Fetch some ticker given the cik with http://yahoo.brand.edgar-online.com/default.aspx?cik=
def fetch_and_insert_tic(cik, connection):
    url = 'http://yahoo.brand.edgar-online.com/default.aspx?cik=' + cik
    response = requests.get(url)
    assert response.status_code == 200
    parsed_html = BeautifulSoup(response.text, 'lxml')

    for span in parsed_html.body.findAll('span'):
        if span.text.startswith('Trade'):
            tic = span.text.split(' ')[1]
            with connection.cursor() as cursor:
                sql = "INSERT INTO `companies` (`cik`, `tic`) VALUES (%s, %s);"
                cursor.execute(sql, (cik, tic))
            connection.commit()
            return True, tic
    return False, None


connection = utils.create_mysql_connection(all_in_mem=True)
db = wrds.Connection()

#data = db.raw_sql('select * from crsp.dsf where permno=12490')
#print(data[data['date'] > datetime.date(year=2016, month=12, day=27)])
# 	prc: A positive amount is an actual close for the trading date while a negative amount denotes the average between BIDLO and ASKHI.
# Prc is the closing price or the negative bid/ask average for a trading day. If the closing price is not available on any given trading day,
# the number in the price field has a negative sign to indicate that it is a bid/ask average and not an actual closing price.
# Please note that in this field the negative sign is a symbol and that the value of the bid/ask average is not negative.

#print(data[data.prc < 0])


sections_to_analyze = [config.DATA_1A_FOLDER]
for section in sections_to_analyze:
    only_cik = 0
    filenames = [x[0].split('/')[-1].split('.')[0] for x in analyze_topics_static.load_and_clean_data(section)]
    cik_2_oindices, already_computed = get_cik_lookup(db, connection)

    # Try to fill database with new TICKERS from CIKS
    for i in range(0, len(filenames)):
        company_cik = filenames[i].split(config.CIK_COMPANY_NAME_SEPARATOR)[0].rjust(10, '0')

        if company_cik not in cik_2_oindices:
            only_cik += 1
            # Try to find tic of other cik
            if not already_computed:
                found, tic = fetch_and_insert_tic(company_cik, connection)

    print("{} reports without any ticker symbol".format(only_cik), '({:.2f}%)'.format(100.0 * float(only_cik) / len(filenames)))

    for filename in filenames:
        company_cik, temp = filename.split(config.CIK_COMPANY_NAME_SEPARATOR)
        company_cik = company_cik.rjust(10, '0')
        submitting_entity_cik, year, internal_number = temp.split('-')  # submitting_entity_cik might be different if it was a third-party filer agent
        year = utils.year_annual_report_comparator(int(year))

connection.close()
