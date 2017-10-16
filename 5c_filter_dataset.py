import os
import pandas as pd
import numpy as np
config = __import__('0_config')


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
    key = 'volatility'
    for section in sections_to_analyze:
        data = pd.read_pickle(os.path.join(section, 'data_df.pkl'))
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data[pd.isnull(data[key]) == False]
        pd.to_pickle(os.path.join(section, 'data_df_filtered.pkl'))

