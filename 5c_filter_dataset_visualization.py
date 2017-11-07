import os
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
config = __import__('0_config')
visualize = __import__('4e_visualize')

if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]

    for section in sections_to_analyze:
        for key in ['volatility', 'roe']:
            input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '_{}.pkl'.format(key))
            data = pd.read_pickle(input_file)
            key_label = key + '_label'
            classes = {x[key_label] for _, x in data.iterrows()}
            mapping = {c:i for i, c in enumerate(sorted(list(classes)))}

            X = [[xx[1] for xx in x['topics']] for _, x in data.iterrows()]
            Y = [mapping[x[key_label]] for _, x in data.iterrows()]

            section = section[2:]
            filename = os.path.join(config.OUTPUT_FOLDER, section.split('/')[-1] + '_' + key)
            model, X = visualize.train_or_load_tsne(filename + '.pkl', X)

            plt.figure()
            for class_val in list(mapping.values()):
                x_ = []
                y_ = []

                for xx, yy in zip(X, Y):
                    if yy == class_val:
                        x_.append(xx)
                        y_.append(yy)

                plt.scatter([xx[0] for xx in x_], [xx[1] for xx in x_])
            plt.legend(list(range(0, len(classes))))
            plt.title('Scatter plot of topics vs {}'.format(key))
            plt.show()

            try:
                years = sorted(list({file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1] for file, _ in data.iterrows()}), key=lambda y:utils.year_annual_report_comparator(int(y)))
            except:
                years = sorted(list({x[1]for x, _ in data.iterrows()}))

            ind = 25 * np.arange(len(classes))  # the x locations for the groups
            width = 0.5

            fig, ax = plt.subplots()
            rects = []
            plt.title('Histogram for ' + key + ' per year')
            plt.xlabel('Label')
            plt.ylabel('Frequency')
            ax.set_xticklabels(tuple(['Label {}'.format(c) for c in classes]), rotation=-90)
            ax.set_xticks(ind + len(classes) * width + len(years) * width / 2)
            for i, y in enumerate(years):
                try:
                    x = [x[key_label] for file, x in data.iterrows() if file.split(config.CIK_COMPANY_NAME_SEPARATOR)[1].split('-')[1] == y]
                except:
                    x = [x[key_label] for index, x in data.iterrows() if index[1] == y]
                vect = [0]*len(classes)
                for xx in x:
                    vect[xx] += 1
                rect = ax.bar(ind + len(classes) * width + i * width, vect, width)
                rects.append(rect)
            if int(years[0]) < 100: # 2 digits
                ax.legend(tuple([rect[0] for rect in rects]), tuple([utils.year_annual_report_comparator(int(y)) for y in years]))
            else:
                ax.legend(tuple([rect[0] for rect in rects]), tuple([y for y in years]))
            plt.show()