import os
import tqdm
from multiprocessing import Process
import random
import utils
process_data = __import__('2a_process_data')
config = __import__('0_config')


def process_folder_multithread(annual_reports):
    connection = utils.create_mysql_connection()

    for annual_report in annual_reports:
        process_folder(annual_report, connection)

    connection.close()


def process_folder(annual_report, connection):
    try:
        buffer = process_data.load_annual_report(annual_report)
    except:
        return

    filename_report = annual_report[annual_report.rfind('/')+1:]
    cik = annual_report[:-len(filename_report)-1]
    cik = cik[cik.rfind('/')+1:]
    info = process_data.extract(buffer, cik, annual_report, '')

    if not info[config.KEY_BADLY_PARSED]:
        # Remove keys which don't need to be updated
        info.pop(config.KEY_CIK, None)
        info.pop(config.KEY_COMPANY_NAME, None)
        info.pop(config.KEY_FILE, None)

        with connection.cursor() as cursor:
            sql = "UPDATE `10k` SET {} WHERE " + config.KEY_FILE + "=%s"
            keys = []
            vals = []
            for key, val in info.items():
                keys.append('`' + key + '`=%s')
                vals.append(val)
            sql = sql.format(' , '.join(keys))
            cursor.execute(sql, tuple(vals + [cik + '/' + filename_report]))

        connection.commit()


if __name__ == "__main__":
    random.seed(0)

    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    connection = process_data.create_mysql_connection()

    bad_parsed_10k_reports = []
    with connection.cursor() as cursor:
        sql = "SELECT (file) FROM `10k` WHERE `{}` = %s;".format(config.KEY_BADLY_PARSED)
        cursor.execute(sql, (True))
        bad_parsed_10k_reports = [os.path.join(config.DATA_AR_FOLDER, row['file']) for row in cursor]

    random.shuffle(bad_parsed_10k_reports) # Better separate work load

    if config.MULTITHREADING:
        connection.close()
        bad_parsed_10k_reports_chunck = process_data.chunks(bad_parsed_10k_reports, 1 + int(len(bad_parsed_10k_reports)/config.NUM_CORES))
        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=process_folder_multithread, args=(bad_parsed_10k_reports_chunck[i],)))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        for annual_report in tqdm.tqdm(bad_parsed_10k_reports, desc="Extract data from badly parsed annual reports"):
            process_folder(annual_report, connection)

        connection.close()
