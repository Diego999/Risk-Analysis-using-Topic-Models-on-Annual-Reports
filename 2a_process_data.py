import os
import glob
import parser_utils
import tqdm
from multiprocessing import Process
import random
import utils
config = __import__('0_config')


def load_annual_report(annual_report):
    buffer = []
    annual_report_clean = annual_report.replace(config.EXTENSION_10K_REPORT, config.EXTENSION_CLEAN_PREPROCESSING)
    with open(annual_report_clean if os.path.exists(annual_report_clean) else annual_report, 'r', encoding='utf-8') as fp:
        for l in fp:
            buffer.append(l)

    if not os.path.exists(annual_report_clean):
        buffer = parser_utils.clean_file(buffer)
        if len(buffer) > 0:
            with open(annual_report_clean, 'w', encoding='utf-8') as fp:
                for l in buffer:
                    fp.write(l + '\n')

    return buffer


def process_folder_multithread(folders):
    connection = utils.create_mysql_connection()

    for folder in folders:
        process_folder(folder, connection)

    connection.close()


def update_key_item(info, val, key1, key2, key3):
    if len(val) > 0:
        info[key1] = val[0]
        if len(val) > 1:
            info[key2] = val[1]
        if len(val) > 2:
            info[key3] = ', '.join([str(v) for v in val[2:]])


def extract(buffer, cik, annual_report, names):
    info = {config.KEY_FILE: cik + '/' + annual_report[annual_report.rfind('/') + 1:], config.KEY_COMPANY_NAME: ', '.join(names), config.KEY_CIK: cik, config.KEY_BADLY_PARSED: False}
    if len(buffer) > 0:
        year_annual_report = annual_report.split('-')[-2]

        # Extract & clean release date
        extracted_release_date = parser_utils.extract_release_date(buffer)
        if extracted_release_date is not None:
            extracted_release_date = parser_utils.clean_date(extracted_release_date, year_annual_report)
            if extracted_release_date is not None:
                info[config.KEY_RELEASED_DATE] = extracted_release_date

        # Extract & clean fiscal year end
        extracted_fiscal_year_end = parser_utils.extract_fiscal_end_year(buffer)
        if extracted_fiscal_year_end is not None:
            extracted_fiscal_year_end = parser_utils.clean_date(extracted_fiscal_year_end, year_annual_report)
            if extracted_fiscal_year_end is not None:
                info[config.KEY_FISCAL_YEAR_END] = extracted_fiscal_year_end
        elif extracted_release_date is not None:  # Infer the date
            extracted_fiscal_year_end = parser_utils.clean_date('31 12 ' + str(year_annual_report), year_annual_report)
            if extracted_fiscal_year_end is not None:
                info[config.KEY_FISCAL_YEAR_END] = extracted_fiscal_year_end

        # Extract beginning of sections
        vals = parser_utils.extract_items(buffer)
        if vals is not None:
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = vals
            update_key_item(info, val_1a, config.KEY_ITEM_1A_1, config.KEY_ITEM_1A_2, config.KEY_ITEM_1A_3)
            update_key_item(info, val_1b, config.KEY_ITEM_1B_1, config.KEY_ITEM_1B_2, config.KEY_ITEM_1B_3)
            update_key_item(info, val_2, config.KEY_ITEM_2_1, config.KEY_ITEM_2_2, config.KEY_ITEM_2_3)
            update_key_item(info, val_7, config.KEY_ITEM_7_1, config.KEY_ITEM_7_2, config.KEY_ITEM_7_3)
            update_key_item(info, val_7a, config.KEY_ITEM_7A_1, config.KEY_ITEM_7A_2, config.KEY_ITEM_7A_3)
            update_key_item(info, val_8, config.KEY_ITEM_8_1, config.KEY_ITEM_8_2, config.KEY_ITEM_8_3)
            update_key_item(info, val_9, config.KEY_ITEM_9_1, config.KEY_ITEM_9_2, config.KEY_ITEM_9_3)
        else:
            info[config.KEY_BADLY_PARSED] = True
    else:
        info[config.KEY_BADLY_PARSED] = True

    return info


def process_folder(folder, connection):
    cik = str(folder[folder.rfind('/') + 1:])
    names = []
    with open(os.path.join(folder, config.NAME_FILE_PER_CIK), 'r', encoding='utf-8') as fp:
        for name in fp:
            names.append(name.strip())
    annual_reports = glob.glob(folder + '/*.' + config.EXTENSION_10K_REPORT)

    for annual_report in reversed(annual_reports):
        try:
            buffer = load_annual_report(annual_report)
        except:
            continue

        info = extract(buffer, cik, annual_report, names)

        with connection.cursor() as cursor:
            sql = "INSERT INTO `10k` ({}) VALUES ({})"
            keys = []
            vals = []
            for key, val in info.items():
                keys.append('`' + key + '`')
                vals.append(val)
            sql = sql.format(', '.join(keys), ', '.join(['%s']*len(keys)))
            cursor.execute(sql, tuple(vals))

    connection.commit()


if __name__ == "__main__":
    random.seed(0)

    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    cik_folders = [os.path.join(config.DATA_AR_FOLDER, d) for d in os.listdir(config.DATA_AR_FOLDER) if os.path.isdir(os.path.join(config.DATA_AR_FOLDER, d))]
    random.shuffle(cik_folders) # Better separate work load

    if config.MULTITHREADING:
        folders = utils.chunks(cik_folders, 1 + int(len(cik_folders)/config.NUM_CORES))
        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=process_folder_multithread, args=(folders[i],)))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        connection = utils.create_mysql_connection()

        for folder in tqdm.tqdm(cik_folders, desc="Extract data from annual reports"):
            process_folder(folder, connection)

        connection.close()
