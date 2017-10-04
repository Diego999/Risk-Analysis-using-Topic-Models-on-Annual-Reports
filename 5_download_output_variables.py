import wrds
import utils
from bs4 import BeautifulSoup
import requests
import os
import re
import pandas as pd
import datetime as dt
import io
import time
import glob
import random
from multiprocessing import Manager, Process
from dateutil.relativedelta import relativedelta
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')

EXTRA_KEYS = ['cusip', 'gvkey', 'lpermno', 'lpermco', 'tic']


def get_cik_lookup_table(db, connection):
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


# From https://github.com/sjev/trading-with-python/blob/master/scratch/get_yahoo_data.ipynb
def stock_get_cookie_and_token():
    url = 'https://uk.finance.yahoo.com/quote/IBM/history'  # url for a ticker symbol, with a download link
    r = requests.get(url)  # download page
    txt = r.text  # extract html

    cookie = r.cookies['B']  # the cooke we're looking for is named 'B'

    # Now we need to extract the token from html.
    # the string we need looks like this: "CrumbStore":{"crumb":"lQHxbbYOBCq"}
    # regular expressions will do the trick!
    pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')

    crumb = None
    for line in txt.splitlines():
        m = pattern.match(line)
        if m is not None:
            crumb = m.groupdict()['crumb']

    return cookie, crumb


# date :(y,m,d)
def get_stock_yahoo(ticker, start_date, end_date, cookie, crumb):
    # prepare input data as a tuple
    data = (ticker,
            int(time.mktime(dt.datetime(*start_date).timetuple())),
            int(time.mktime(dt.datetime(*end_date).timetuple())),
            crumb)

    # Prepare URL
    url = "https://query1.finance.yahoo.com/v7/finance/download/{0}?period1={1}&period2={2}&interval=1d&events=history&crumb={3}".format(*data)
    data = requests.get(url, cookies={'B': cookie})

    # Fetch data
    buf = io.StringIO(data.text)  # create a buffer
    res = None
    try:
        df = pd.read_csv(buf, index_col=0)  # convert to pandas DataFrame
        if 'Close' in df:
            res =  df['Close'].to_dict() # {Date:Close price}
    except:
        pass

    return res


def get_stock_crsp(db, key, index, start_date, end_date):
    data = db.raw_sql('select * from crsp.dsf where ' + key + '=\'' + index + '\' and date(crsp.dsf.date) between date(\'' + start_date + '\') and date(\'' + end_date + '\')')
    # 	prc: A positive amount is an actual close for the trading date while a negative amount denotes the average between BIDLO and ASKHI.
    # Prc is the closing price or the negative bid/ask average for a trading day. If the closing price is not available on any given trading day,
    # the number in the price field has a negative sign to indicate that it is a bid/ask average and not an actual closing price.
    # Please note that in this field the negative sign is a symbol and that the value of the bid/ask average is not negative.

    stock = None
    if len(data) > 0:  # We've found some stocks
        stock = {date.strftime('%Y-%m-%d'):prc for date, prc in data.set_index('date')['prc'].to_dict().items()}  # Convert to {date:closed price}
    return stock


def get_net_income_and_stockholder_equity(db, key, val):
    # ni gives net income and seq/teq approximately the same thing for stockholder equity. Divide both to obtain return of equity
    #data = db.raw_sql("select gvkey, tic, cusip, cik, ni, seq, teq, datadate, fyear from compa.funda where key = "'' + val + "'")
    data = db.raw_sql("select distinct ni, seq, teq, datadate, fyear from compa.funda where " + key + " = '" + val + "' and ni != 'NaN' and (seq != 'NaN' or teq != 'NaN')")
    res = None
    if len(data) > 0: # We've found net income/seq
        res = {date.strftime('%Y-%m-%d'):vals for date, vals in data.set_index('datadate')[['ni', 'seq', 'teq']].T.to_dict().items()}
    return res


def get_keys_with_net_income_and_stockholder_equity(db, cik):
    data = db.raw_sql("select distinct tic, cusip, cik from compa.funda where cik = '" + cik + "'")
    res = {}
    if len(data) > 0:
        res = data.set_index(['cik']).T.to_dict()
    return res


def gather_stock(filename, tickers, cusips, lpermnos, lpermcos, connection, db, cookie, crumb):
    stocks = []
    start_date = None
    end_date = None
    fiscal_year_end = None

    # Gather release_date to compute the start and end date (1 year after)
    mysql_key = filename.replace(config.CIK_COMPANY_NAME_SEPARATOR, '/') + '.txt'
    with connection.cursor() as cursor:
        sql = "SELECT `release_date`, `fiscal_year_end` from `10k` WHERE `file` = %s;"
        cursor.execute(sql, mysql_key)
        rows = cursor.fetchall()
        assert len(rows) == 1
        start_date, fiscal_year_end = rows[0]['release_date'], rows[0]['fiscal_year_end']

    if start_date is None and fiscal_year_end is not None:
        start_date = fiscal_year_end  # Happens only for old companies, 1994-12-31

    if start_date is not None:
        # First try with other indices from CRSP lookup-table
        indices = {'cusip': cusips, 'permno': lpermnos, 'permco': lpermcos}
        start_date_others = start_date.strftime('%Y-%m-%d')
        end_date_others = (start_date + relativedelta(years=1)).strftime('%Y-%m-%d')  # add only 1 year without extra day because MYSQL include the last day in the range
        for key, indices in indices.items():
            for index in indices:
                stock = get_stock_crsp(db, key, index, start_date_others, end_date_others)
                if stock is not None:
                    stocks.append(stock)

        # If still no stocks, try to look up stocks using tickers
        if len(stocks) == 0:
            for ticker in tickers:
                # Compute final start/end date
                end_date_tic = start_date + relativedelta(days=1, years=1)  # add year + 1 day, because the API return up to endDate (not included)
                start_date_tic = (start_date.year, start_date.month, start_date.day)
                end_date_tic = (end_date_tic.year, end_date_tic.month, end_date_tic.day)
                stock = get_stock_yahoo(ticker, start_date_tic, end_date_tic, cookie, crumb)
                if stock is not None:
                    stocks.append(stock)

    return stocks


def gather_net_income_and_stockholder_equity(company_cik, tickers, cusips, lpermnos, lpermcos, connection, db):
    ni_seqs = []
    res = get_net_income_and_stockholder_equity(db, 'cik', company_cik)
    if res is not None:
        ni_seqs.append(res)

        unknown_indices = len(tickers) == 0 and len(cusips) == 0 and len(lpermnos) == 0 and len(lpermcos) == 0
        incomplete_indices = len(tickers) == 0 or len(cusips) == 0
        # Add other keys
        if unknown_indices or incomplete_indices:
            keys = get_keys_with_net_income_and_stockholder_equity(db, company_cik)[company_cik]
            with connection.cursor() as cursor:
                if unknown_indices: # Add to database new values
                    sql = "INSERT INTO `companies` ({}) VALUES ({})"
                    sql_fina_keys = ['cik']
                    sql_final_values = [company_cik]
                    if len(tickers) == 0:
                        sql_fina_keys += ['tic']
                        sql_final_values += [keys['tic']]
                    if len(cusips) == 0:
                        sql_fina_keys += ['cusip']
                        sql_final_values += [keys['cusip']]
                    sql = sql.format(', '.join(sql_fina_keys), ', '.join(['%s'] * len(sql_fina_keys)))
                    cursor.execute(sql, tuple(sql_final_values))
                elif incomplete_indices: # Update database
                    if keys['tic'] not in tickers:
                        tickers.add(keys['tic'])
                        sql = 'UPDATE companies SET tic = %s WHERE cik = %s;'
                        cursor.execute(sql, (', '.join(tickers), company_cik))
                    if keys['cusip'] not in cusips:
                        cusips.add(keys['cusip'])
                        sql = 'UPDATE companies SET cusip = %s WHERE cik = %s;'
                        cursor.execute(sql, (', '.join(cusips), company_cik))
            connection.commit()

    # Try also with other indices from CRSP lookup-table
    indices = {'cusip': cusips, 'tic': tickers}
    for key, indices in indices.items():
        for index in indices:
            res = get_net_income_and_stockholder_equity(db, key, index)
            if res is not None:
                ni_seqs.append(res)

    return ni_seqs


def compute_process(filenames, cik_2_oindices, stocks_already_computed, ni_seqs_already_computed, cookie, crumb, error_stocks, error_ni_seqs, pid):
    connection = utils.create_mysql_connection(all_in_mem=True)
    db = wrds.Connection()

    for filename in filenames:
        compute_process_utils(filename, cik_2_oindices, stocks_already_computed, ni_seqs_already_computed, connection, db, cookie, crumb, error_stocks, error_ni_seqs, pid)

    connection.close()


def compute_process_utils(filename, cik_2_oindices, stocks_already_computed, ni_seqs_already_computed, connection, db, cookie, crumb, error_stocks, error_ni_seqs, pid):
    company_cik, temp = filename.split(config.CIK_COMPANY_NAME_SEPARATOR)
    company_cik = company_cik.rjust(10, '0')
    submitting_entity_cik, year, internal_number = temp.split('-')  # submitting_entity_cik might be different if it was a third-party filer agent
    year = utils.year_annual_report_comparator(int(year))

    # If there exists at least one TICKER
    tickers, cusips, lpermnos, lpermcos = set(), set(), set(), set()
    if company_cik in cik_2_oindices:
        tickers = cik_2_oindices[company_cik]['tic']  # Always present
        if 'cusip' in cik_2_oindices[company_cik]:
            cusips = cik_2_oindices[company_cik]['cusip']
        if 'lpermno' in cik_2_oindices[company_cik]:
            lpermnos = cik_2_oindices[company_cik]['lpermno']
        if 'lpermco' in cik_2_oindices[company_cik]:
            lpermcos = cik_2_oindices[company_cik]['lpermco']

    # Get net income & stockholder's equity
    error_ni_seqs_utils = []
    if filename not in ni_seqs_already_computed:
        ni_seqs = gather_net_income_and_stockholder_equity(company_cik, tickers, cusips, lpermnos, lpermcos, connection, db)
        if len(ni_seqs) > 0:
            output = os.path.join(config.DATA_NI_SEQ_FOLDER, filename) + '.pkl'
            utils.save_pickle(ni_seqs, output)
        else:
            error_ni_seqs_utils.append(filename)
    error_ni_seqs[pid] += error_ni_seqs_utils

    # Get stock prices
    error_stocks_utils = []
    if filename not in stocks_already_computed:
        stocks = []
        # If there is at least one index
        if sum([len(x) for x in [tickers, cusips, lpermcos, lpermnos]]) > 0:
            stocks = gather_stock(filename, tickers, cusips, lpermnos, lpermcos, connection, db, cookie, crumb)

        if len(stocks) > 0:
            output = os.path.join(config.DATA_STOCKS_FOLDER, filename) + '.pkl'
            utils.save_pickle(stocks, output)
        else:
            error_stocks_utils.append(filename)
    error_stocks[pid] += error_stocks_utils


if not os.path.exists(config.DATA_STOCKS_FOLDER):
    os.makedirs(config.DATA_STOCKS_FOLDER)
if not os.path.exists(config.DATA_NI_SEQ_FOLDER):
    os.makedirs(config.DATA_NI_SEQ_FOLDER)

sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
for section in sections_to_analyze:
    print(section)
    stocks_already_computed = {f.split('/')[-1][:-4] for f in glob.glob("{}/*.pkl".format(config.DATA_STOCKS_FOLDER))}
    ni_seqs_already_computed = {f.split('/')[-1][:-4] for f in glob.glob("{}/*.pkl".format(config.DATA_NI_SEQ_FOLDER))}

    connection = utils.create_mysql_connection(all_in_mem=True)
    db = wrds.Connection()
    cik_2_oindices, already_computed = get_cik_lookup_table(db, connection)
    connection.close()

    # Try to fill database with new TICKERs from CIKs
    only_cik = 0
    filenames = [x[0].split('/')[-1].split('.')[0] for x in analyze_topics_static.load_and_clean_data(section)]
    random.shuffle(filenames)  # Better balanced work for multiprocessing

    for i in range(0, len(filenames)):
        company_cik = filenames[i].split(config.CIK_COMPANY_NAME_SEPARATOR)[0].rjust(10, '0')
        if company_cik not in cik_2_oindices:
            only_cik += 1
            # Try to find tic of other cik
            if not already_computed:
                found, tic = fetch_and_insert_tic(company_cik, connection)
    
    print("{} reports without any ticker symbol".format(only_cik), '({:.2f}%)'.format(100.0 * float(only_cik) / len(filenames)))

    cookie, crumb = stock_get_cookie_and_token()

    manager = Manager()
    if config.MULTITHREADING:
        filenames_chunk = utils.chunks(filenames, 1 + int(len(filenames) / config.NUM_CORES))
        error_stocks = manager.list([[]] * config.NUM_CORES)
        error_ni_seqs = manager.list([[]] * config.NUM_CORES)

        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=compute_process, args=(filenames_chunk[i], cik_2_oindices, stocks_already_computed, ni_seqs_already_computed, cookie, crumb, error_stocks, error_ni_seqs, i)))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        error_stocks = [[]]
        error_ni_seqs = [[]]
        compute_process(filenames, cik_2_oindices, stocks_already_computed, ni_seqs_already_computed, cookie, crumb, error_stocks, error_ni_seqs, 0)

    # Reduce
    error_stocks = sum(error_stocks, [])
    error_ni_seqs = sum(error_ni_seqs, [])

    # Write backs filenames which obtained an error
    if len(error_stocks) > 0:
        with open('stock_error_{}.txt'.format(section[section.rfind('/') + 1:]), 'w') as fp_stocks:
            for f in error_stocks:
                fp_stocks.write(f + '\n')
    if len(error_ni_seqs) > 0:
        with open('ni_seq_error_{}.txt'.format(section[section.rfind('/') + 1:]), 'w') as fp_ni_seqs:
            for f in error_ni_seqs:
                fp_ni_seqs.write(f + '\n')
