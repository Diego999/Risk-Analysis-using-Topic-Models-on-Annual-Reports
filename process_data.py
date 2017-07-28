import os
import config
import glob
import parser_utils
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import random


def process_folder(folder):
    cik = str(folder[folder.rfind('/') + 1:])
    names = []
    with open(os.path.join(folder, config.NAME_FILE_PER_CIK), 'r', encoding='utf-8') as fp:
        for name in fp:
            names.append(name.strip())
    annual_reports = glob.glob(folder + '/*.txt')

    for annual_report in reversed(annual_reports):
        buffer = []
        info = {config.KEY_COMPANY_NAME: names, config.KEY_CIK: cik}
        with open(annual_report, 'r', encoding='utf-8') as fp:
            buffer = parser_utils.clean_file(fp.readlines())
        year_annual_report = annual_report.split('-')[-2]

        # Extract & clean release date
        extracted_release_date = parser_utils.extract_release_date(buffer)
        if extracted_release_date is not None:
            info[config.KEY_RELEASED_DATE] = parser_utils.clean_date(extracted_release_date, year_annual_report)
        else:
            with open(config.LOG_RELEASE_DATE_MISSING, 'a', encoding='utf-8') as fp:
                fp.write(annual_report + '\n')

        # Extract & clean fiscal year end
        extracted_fiscal_year_end = parser_utils.extract_fiscal_end_year(buffer)
        if extracted_fiscal_year_end is not None:
            info[config.KEY_FISCAL_YEAR_END] = parser_utils.clean_date(extracted_fiscal_year_end, year_annual_report)
        else:
            prefix = ''
            # Infer the date
            if extracted_release_date is not None:
                prefix = 'FIXED '
                info[config.KEY_FISCAL_YEAR_END] = parser_utils.clean_date('31 12 ' + str(year_annual_report))

            with open(config.LOG_FISCAL_YEAR_END_MISSING, 'a', encoding='utf-8') as fp:
                fp.write(prefix + annual_report + '\n')

if __name__ == "__main__":
    random.seed(0)

    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    cik_folders = [d for d in os.listdir(config.DATA_AR_FOLDER) if os.path.isdir(os.path.join(config.DATA_AR_FOLDER, d))]
    random.shuffle(cik_folders) # Better separate work load

    if config.MULTITHREADING:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(process_folder)(folder) for folder in cik_folders)
    else:
        for i, folder in tqdm.tqdm(enumerate(cik_folders), desc="Extract data from annual reports"):
            process_folder(folder)
