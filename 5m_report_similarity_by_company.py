import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import utils
import matplotlib.pyplot as plt
import gensim
config = __import__('0_config')


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]

    for section in sections_to_analyze:
        input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '.pkl')
        data = pd.read_pickle(input_file)

        data_list = []
        for file, values in data.iterrows():
            values = values.to_dict()
            values['file'] = file
            values['cik'] = file.split(config.CIK_COMPANY_NAME_SEPARATOR)[0]
            values['year'] = utils.year_annual_report_comparator(int(file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1]))
            data_list.append(values)

        data_list = sorted(data_list, key=lambda x: (x['cik'], x['year']))
        data_list_reports_by_company_sorted_by_year = {}

        for i in range(1, len(data_list)):
            x_previous = data_list[i-1]
            x_current = data_list[i]

            if x_previous['cik'] == x_current['cik']:
                if x_current['year'] == (x_previous['year'] + 1):
                    if x_current['cik'] not in data_list_reports_by_company_sorted_by_year:
                        data_list_reports_by_company_sorted_by_year[x_current['cik']] = set()
                    data_list_reports_by_company_sorted_by_year[x_current['cik']].add((x_previous['file'], x_previous['year']))
                    data_list_reports_by_company_sorted_by_year[x_current['cik']].add((x_current['file'], x_current['year']))

        for k in data_list_reports_by_company_sorted_by_year.keys():
            data_list_reports_by_company_sorted_by_year[k] = sorted(list(data_list_reports_by_company_sorted_by_year[k]), key=lambda x:x[1])

        global_sims = []
        for cik, files_years in data_list_reports_by_company_sorted_by_year.items():
            #print(cik)
            sims = []
            for i in range(1, len(files_years)):
                previous_file, previous_year = files_years[i-1]
                current_file, current_year = files_years[i]

                if current_year == (previous_year + 1):
                    previous_text = data.loc[previous_file]['text'].lower()
                    current_text = data.loc[current_file]['text'].lower()

                    docs = [previous_text, current_text]
                    vect = TfidfVectorizer(min_df=1)
                    tfidf = vect.fit_transform(docs)
                    sim = (tfidf * tfidf.T).A[0][1]
                    sims.append(sim)
            global_sims.append(sims)
            #print('\t' + ' '.join(['{:.2f}%'.format(sim) for sim in sims]))

        all_sims = sum(global_sims, [])

        plt.figure()
        plt.boxplot(all_sims)

        plt.figure()
        plt.hist(all_sims, bins=1000)

        plt.show()