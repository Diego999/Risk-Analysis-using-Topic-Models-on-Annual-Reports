import os
import glob
import matplotlib.pyplot as plt
import numpy as np
config = __import__('0_config')


if __name__ == "__main__":
    results = glob.glob(os.path.join(config.OUTPUT_FOLDER, 'topics') + '/*.txt')
    parsed_data = []
    for r in results:
        vals = r[:r.rfind('.')].split('_')

        t = {}
        for v in vals:
            key = ''
            if v.startswith('cu'):
                key = 'cu'
            elif v.startswith('cv'):
                key = 'cv'
            elif v.startswith('topics'):
                key = 'topics'

            if key != '':
                t[key] = float(v.split(':')[1])
        parsed_data.append(t)

    parsed_data = sorted(parsed_data, key=lambda t:int(t['topics']))
    num_topics = [int(t['topics']) for t in parsed_data]
    x = range(min(num_topics), max(num_topics)+1)

    cu = [float(t['cu']) for t in parsed_data]
    cv = [float(t['cv']) for t in parsed_data]
    idx_max_cu, max_cu = np.argmax(cu) + min(num_topics), max(cu)
    idx_max_cv, max_cv = np.argmax(cv) + min(num_topics), max(cv)

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x, cv, 'b')
    plt.plot(idx_max_cv, max_cv, 'ro')
    plt.legend(('c_v', 'best c_v ({})'.format(idx_max_cv)), loc='best')
    plt.xlabel('number of topics')
    plt.ylabel('Coherence score (CV)')
    plt.xticks(np.arange(min(x)-1, max(x) + 1, 5.0))

    plt.subplot(212)
    plt.plot(x, cu, 'r')
    plt.plot(idx_max_cu, max_cu, 'bo')
    plt.legend(('c_u', 'best c_u ({})'.format(idx_max_cu)), loc='best')
    plt.xlabel('number of topics')
    plt.ylabel('Coherence score (CU)')
    plt.xticks(np.arange(min(x)-1, max(x) + 1, 5.0))

    plt.show()