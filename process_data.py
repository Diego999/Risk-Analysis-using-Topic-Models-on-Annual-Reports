import os
import config
import glob
import parser_utils
import tqdm


if __name__ == "__main__":
    if not os.path.isdir(config.DATA_AR_FOLDER):
        print('ERR: {} does not exist'.format(config.DATA_AR_FOLDER))

    cik_folders = [x[0] for x in os.walk(config.DATA_AR_FOLDER)][1:]
    for folder in tqdm.tqdm(reversed(cik_folders), desc='Extracting fiscal year end'):
        cik = str(folder[folder.rfind('/') + 1:])
        names = []
        with open(os.path.join(folder, config.NAME_FILE_PER_CIK), 'r', encoding='utf-8') as fp:
            for name in fp:
                names.append(name.strip())
        annual_reports = glob.glob(folder + '/*.txt')

        for annual_report in reversed(annual_reports):
            buffer = []
            info = {config.KEY_COMPANY_NAME:names, config.KEY_CIK:cik}
            with open(annual_report, 'r', encoding='utf-8') as fp:
                buffer = fp.readlines()

            # Extract & clean fiscal year end
            year_annual_report = annual_report.split('-')[-2]
            info[config.KEY_FISCAL_YEAR_END] = parser_utils.clean_fiscal_year_end(parser_utils.extract_fiscal_end_year(buffer), year_annual_report)
            assert info[config.KEY_FISCAL_YEAR_END] is not None

