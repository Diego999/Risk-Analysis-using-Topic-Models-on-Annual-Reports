import utils
import os
import glob
import numpy as np
import math
analyze_topics_static = __import__('4a_analyze_topics_static')
infer_topics = __import__('4d_infer_topics')
config = __import__('0_config')

stocks = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_STOCKS_FOLDER))}
ni_seqs = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_NI_SEQ_FOLDER))}
sec_inds = {f.split('/')[-1][:-4]:f for f in glob.glob("{}/*.pkl".format(config.DATA_SEC_IND_FOLDER))}
if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]
    for section in sections_to_analyze:
        data = analyze_topics_static.load_and_clean_data(section)