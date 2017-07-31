import os
import config
import glob
import parser_utils
import tqdm
from joblib import Parallel, delayed
import random


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


def process_folder(folder):
    cik = str(folder[folder.rfind('/') + 1:])
    names = []
    with open(os.path.join(folder, config.NAME_FILE_PER_CIK), 'r', encoding='utf-8') as fp:
        for name in fp:
            names.append(name.strip())
    annual_reports = glob.glob(folder + '/*.' + config.EXTENSION_10K_REPORT)

    for annual_report in reversed(annual_reports):
        buffer = load_annual_report(annual_report)
        flag_bad_parsing = False
        if len(buffer) > 0:
            info = {config.KEY_COMPANY_NAME: names, config.KEY_CIK: cik}
            year_annual_report = annual_report.split('-')[-2]

            # Extract & clean release date
            extracted_release_date = parser_utils.extract_release_date(buffer)
            info[config.KEY_RELEASED_DATE] = parser_utils.clean_date(extracted_release_date, year_annual_report)

            # Extract & clean fiscal year end
            extracted_fiscal_year_end = parser_utils.extract_fiscal_end_year(buffer)
            if extracted_fiscal_year_end is not None:
                info[config.KEY_FISCAL_YEAR_END] = parser_utils.clean_date(extracted_fiscal_year_end, year_annual_report)
            elif extracted_release_date is not None: # Infer the date
                info[config.KEY_FISCAL_YEAR_END] = parser_utils.clean_date('31 12 ' + str(year_annual_report))

            # Extract beginning of sections
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = [], [], [], [], [], [], []
            vals = parser_utils.extract_items(buffer)
            if vals is not None:
                val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = vals
            else:
                flag_bad_parsing = True
        else:
            flag_bad_parsing = True

        # TODO Input data in MySQL
        if flag_bad_parsing:
            print(flag_bad_parsing, extracted_release_date, extracted_fiscal_year_end, val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9)


if __name__ == "__main__":
    random.seed(0)

    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    cik_folders = [os.path.join(config.DATA_AR_FOLDER, d) for d in os.listdir(config.DATA_AR_FOLDER) if os.path.isdir(os.path.join(config.DATA_AR_FOLDER, d))]
    random.shuffle(cik_folders) # Better separate work load

    if config.MULTITHREADING:
        Parallel(n_jobs=config.NUM_CORES)(delayed(process_folder)(folder) for folder in cik_folders)
    else:
        for folder in tqdm.tqdm(cik_folders, desc="Extract data from annual reports"):
            process_folder(folder)
