import os
import pandas as pd
import numpy as np
config = __import__('0_config')


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
    keys = ['volatility', 'roe']
    for section in sections_to_analyze:
        for key in keys:
            input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '.pkl')
            data = pd.read_pickle(input_file)
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data[pd.isnull(data[key]) == False]
            output_file = input_file.replace(config.SUFFIX_DF, config.SUFFIX_DF + key)
            pd.to_pickle(data, output_file)

