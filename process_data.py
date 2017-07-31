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
        # Extract beginning of sections
        val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = parser_utils.extract_items(buffer)

        # Check if there is only 1 occurence starting by "Item XXX" or "Title"
        val_1a = parser_utils.simplify_occurences(val_1a, config.KEY_WORDS_ITEM_1A, buffer)
        val_1b = parser_utils.simplify_occurences(val_1b, config.KEY_WORDS_ITEM_1B, buffer)
        val_2 = parser_utils.simplify_occurences(val_2, config.KEY_WORDS_ITEM_2, buffer)
        val_7 = parser_utils.simplify_occurences(val_7, config.KEY_WORDS_ITEM_7, buffer)
        val_7a = parser_utils.simplify_occurences(val_7a, config.KEY_WORDS_ITEM_7A, buffer)
        val_8 = parser_utils.simplify_occurences(val_8, config.KEY_WORDS_ITEM_8, buffer)
        val_9 = parser_utils.simplify_occurences(val_9, config.KEY_WORDS_ITEM_9, buffer)

        if sum([(1 if len(l) > 1 else 0) for l in [val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9]]) != 0: # Not a unique number per items
            total_length = 0
            while total_length != len(val_1a + val_1b + val_2 + val_7 + val_7a + val_8 + val_9):
                total_length = len(val_1a + val_1b + val_2 + val_7 + val_7a + val_8 + val_9)

                # Remove table of contents refs and before
                val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = parser_utils.remove_toc_and_before([val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])

                # If there are still several indices, let's try to disambiguate them
                val_1a = parser_utils.disambiguate(val_1a, config.KEY_WORDS_ITEM_1A, buffer)
                val_1b = parser_utils.disambiguate(val_1b, config.KEY_WORDS_ITEM_1B, buffer)
                val_2 = parser_utils.disambiguate(val_2, config.KEY_WORDS_ITEM_2, buffer)
                val_7 = parser_utils.disambiguate(val_7, config.KEY_WORDS_ITEM_7, buffer)
                val_7a = parser_utils.disambiguate(val_7a, config.KEY_WORDS_ITEM_7A, buffer)
                val_8 = parser_utils.disambiguate(val_8, config.KEY_WORDS_ITEM_8, buffer)
                val_9 = parser_utils.disambiguate(val_9, config.KEY_WORDS_ITEM_9, buffer)

                # Remove continuous elements to keep only the first one
                val_1a = parser_utils.remove_continuous(val_1a)
                val_1b = parser_utils.remove_continuous(val_1b)
                val_2 = parser_utils.remove_continuous(val_2)
                val_7 = parser_utils.remove_continuous(val_7)
                val_7a = parser_utils.remove_continuous(val_7a)
                val_8 = parser_utils.remove_continuous(val_8)
                val_9 = parser_utils.remove_continuous(val_9)

        # If there are still more indices per item
        if len(val_1a) > 1 or len(val_1b) > 1 or len(val_2) > 1 or len(val_7) > 1 or len(val_7a) > 1 or len(val_8) > 1 or len(val_9) > 1:
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = parser_utils.remove_first_ref_previous_item([val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = parser_utils.remove_last_ref_next_item([val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])


            with open(config.LOG_FISCAL_YEAR_END_MISSING, 'a', encoding='utf-8') as fp:
                fp.write(prefix + annual_report + '\n')

if __name__ == "__main__":
    random.seed(0)

    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    cik_folders = [os.path.join(config.DATA_AR_FOLDER, d) for d in os.listdir(config.DATA_AR_FOLDER) if os.path.isdir(os.path.join(config.DATA_AR_FOLDER, d))]
    random.shuffle(cik_folders) # Better separate work load

    if config.MULTITHREADING:
        Parallel(n_jobs=config.NUM_CORES)(delayed(process_folder)(folder) for folder in cik_folders)
    else:
        for i, folder in tqdm.tqdm(enumerate(cik_folders), desc="Extract data from annual reports"):
            process_folder(folder)
