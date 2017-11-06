import os
import pandas as pd
import numpy as np
import utils
config = __import__('0_config')


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]

    for section in sections_to_analyze:
        input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '.pkl')
        data = pd.read_pickle(input_file)
        data = data.replace([np.inf, -np.inf], np.nan)

        key = 'volatility'
        data = data[pd.isnull(data[key]) == False]
        output_file = input_file.replace(config.SUFFIX_DF, config.SUFFIX_DF + '_' + key)
        pd.to_pickle(data, output_file)

        key = 'roe'
        data = data[pd.isnull(data[key]) == False]

        data_list = []
        for file, values in data.iterrows():
            values = values.to_dict()
            values['file'] = file
            values['cik'] = file.split(config.CIK_COMPANY_NAME_SEPARATOR)[0]
            values['year'] =  utils.year_annual_report_comparator(int(file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1]))
            data_list.append(values)

        data_list = sorted(data_list, key=lambda x: (x['cik'], x['year']))
        data_list_with_roe_next_year = []

        for i in range(1, len(data_list)):
            x_previous = data_list[i-1]
            x_current = data_list[i]

            if x_previous['cik'] == x_current['cik']:
                if x_current['year'] == (x_previous['year'] + 1):
                    x_previous['roe_t+1'] = x_current['roe']
                    x_previous['roe_t+1_relative'] = (x_previous['roe_t+1'] - x_previous['roe'])/abs(x_previous['roe'] + 1e-10)
                    data_list_with_roe_next_year.append(x_previous)

        data_with_roe_next_year = pd.DataFrame(data_list_with_roe_next_year)
        data_with_roe_next_year.set_index(['cik', 'year', 'file'], inplace=True)

        output_file = input_file.replace(config.SUFFIX_DF, config.SUFFIX_DF + '_' + key)
        pd.to_pickle(data_with_roe_next_year, output_file)





