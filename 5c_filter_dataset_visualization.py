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
        for key in ['volatility']:#, 'roe']:
            input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '_{}.pkl'.format(key))
            data = pd.read_pickle(input_file)
            key_label = key + '_label'
            classes = {x[key_label] for _, x in data.iterrows()}
            mapping = {c:i for i, c in enumerate(sorted(list(classes)))}

            X = [[xx[1] for xx in x['topics']] for _, x in data.iterrows()]
            Y = [mapping[x[key_label]] for _, x in data.iterrows()]

            section = section[2:]
            filename = os.path.join(config.OUTPUT_FOLDER, section.split('/')[-1] + '_' + key)
            model, proj_lda = visualize.train_or_load_tsne(filename + '.pkl', X)

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