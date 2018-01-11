import numpy as np
import utils
analyze_topics_static = __import__('4a_analyze_topics_static')
config = __import__('0_config')

if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER]

    for section in sections_to_analyze:
        print(section)
        data = analyze_topics_static.load_and_clean_data(section)

        years = {}
        for filename, text in data:
            year = utils.year_annual_report_comparator(utils.extract_year_from_filename_annual_report(filename))
            if year not in years:
                years[year] = []
            years[year].append(len(text.split()))

        for y, l in years.items():
            years[y] = {'sum':np.sum(l), 'mean':np.mean(l), 'nb_docs':len(l)}

        total = {'sum':0, 'mean':0, 'nb_docs':0}
        for y, vals in sorted(years.items(), key=lambda x:x[0]):
            print(y, '&', '{:,.1f}K'.format(vals['sum']/1000.0), '&', '{:,}'.format(vals['nb_docs']), '&', '{:,}'.format(int(vals['mean'])), '\\\\')
            for k, v in vals.items():
                total[k] += v
        print('\\hline')
        print('Total', '&', '{:,.1f}K'.format(total['sum']/1000.0), '&', '{:,}'.format(total['nb_docs']), '&', '{:,}'.format(int(total['mean']/len(years))), '\\\\')
        print('\\hline')