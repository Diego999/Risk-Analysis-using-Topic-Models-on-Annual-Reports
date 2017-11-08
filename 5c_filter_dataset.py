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
        data_list = []
        for file, values in data.iterrows():
            values['file'] = file
            data_list.append(values)

        # As in On the risk prediction and analysis of soft information in finance reports
        # Too gaussian, classes without labels !
        all_volatilites = data['volatility'].tolist()
        m = np.mean(all_volatilites)
        s = np.std(all_volatilites)
        u = 1
        l = 2
        #ranges = [[-np.inf, m - 2 * s], [m - 2 * s + 1e-10, m - s], [m - s + 1e-10, m + s], [m + s + 1e-10, m + 2 * s], [m + 2 * s + 1e-10, np.inf]]
        ranges = [[-np.inf, m - s], [m - s + 1e-10, m + s], [m + s + 1e-10, np.inf]]
        #counters = [0, 0, 0, 0, 0]
        #for v in all_volatilites:
        #    for i, (r1, r2) in enumerate(ranges):
        #        if r1 <= v <= r2:
        #            counters[i] += 1
        #            break
        for x in data_list:
            for i, (r1, r2) in enumerate(ranges):
                if r1 <= x['volatility'] <= r2:
                    x['volatility_label'] = i
            assert 'volatility_label' in x

        # Uniform split
        #k = 3 # nb classes
        #offset = int(len(data_list)/k)
        #data_list = sorted(data_list, key=lambda x:x['volatility'])
        #for k, i in enumerate(range(0, len(data_list), offset)):
        #    for j in range(i, min(i+offset, len(data_list))):
        #        data_list[j]['volatility_label'] = k

        data_with_vol_labels = pd.DataFrame(data_list)
        data_with_vol_labels.set_index(['file'], inplace=True)
        output_file = input_file.replace(config.SUFFIX_DF, config.SUFFIX_DF + '_' + key)
        pd.to_pickle(data_with_vol_labels, output_file)

        key = 'roe'
        roe_threshold = 5/100.0
        data = data[pd.isnull(data[key]) == False]

        data_list = []
        for file, values in data.iterrows():
            values = values.to_dict()
            values['file'] = file
            values['cik'] = file.split(config.CIK_COMPANY_NAME_SEPARATOR)[0]
            values['year'] = utils.year_annual_report_comparator(int(file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1]))
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
                    x_previous['roe_label'] = (1 if x_previous['roe_t+1_relative'] >= roe_threshold else (-1 if x_previous['roe_t+1_relative'] <= -roe_threshold else 0))
                    data_list_with_roe_next_year.append(x_previous)

        data_with_roe_next_year = pd.DataFrame(data_list_with_roe_next_year)
        data_with_roe_next_year.set_index(['cik', 'year', 'file'], inplace=True)

        output_file = input_file.replace(config.SUFFIX_DF, config.SUFFIX_DF + '_' + key)
        pd.to_pickle(data_with_roe_next_year, output_file)





