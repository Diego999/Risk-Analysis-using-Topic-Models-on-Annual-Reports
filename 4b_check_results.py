import os
import glob
import matplotlib.pyplot as plt
import numpy as np
config = __import__('0_config')


def check_param_topic():
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

    parsed_data = sorted(parsed_data, key=lambda t: int(t['topics']))
    num_topics = [int(t['topics']) for t in parsed_data]
    x = range(min(num_topics), max(num_topics) + 1)

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
    plt.xticks(np.arange(min(x) - 1, max(x) + 1, 5.0))

    plt.subplot(212)
    plt.plot(x, cu, 'r')
    plt.plot(idx_max_cu, max_cu, 'bo')
    plt.legend(('c_u', 'best c_u ({})'.format(idx_max_cu)), loc='best')
    plt.xlabel('number of topics')
    plt.ylabel('Coherence score (CU)')
    plt.xticks(np.arange(min(x) - 1, max(x) + 1, 5.0))

    plt.show()


def check_tuning(k=3):
    results = glob.glob(config.OUTPUT_FOLDER + '/*.txt')
    parsed_data = []
    for r in results:
        vals = r[:r.rfind('.')].split('_')
        t = {k:v for k,v in [v.split(':') for v in vals if ':' in v]}
        parsed_data.append(t)

    parsed_data_sorted_by_cu = sorted(parsed_data, key=lambda x:-float(x['cu']))
    parsed_data_sorted_by_cv = sorted(parsed_data, key=lambda x:-float(x['cv']))

    for i, (cu, cv) in enumerate(zip(parsed_data_sorted_by_cu[:k], parsed_data_sorted_by_cv[:k])):
        cu_sorted = sorted(cu.items(), key=lambda x:x[0])
        cv_sorted = sorted(cv.items(), key=lambda x:x[0])

        print('Best #{} (cu, cv)'.format(i+1))
        for a, b in zip(cu_sorted, cv_sorted):
            assert a[0] == b[0]
            print('\t', '\t', a[0], '\t', '\t', round(a[1], 4) if not isinstance(a[1], str) else a[1], '\t', '\t', round(b[1], 4) if not isinstance(b[1], str) else b[1])


if __name__ == "__main__":
    check_param_topic()
    check_tuning()