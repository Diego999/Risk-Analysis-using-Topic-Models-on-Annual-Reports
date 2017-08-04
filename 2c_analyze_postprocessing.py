import numpy as np
import utils
config = __import__('0_config')
process_data = __import__('2a_process_data')


def hist(vals, bin_width):
    bins = [i for i in range(0, np.max(vals), bin_width)]
    freqs, bins = np.histogram(vals, bins=bins)
    freqs = list(freqs)
    bins = list(bins)
    norm_freqs = [100.0*float(freq)/len(vals) for freq in freqs]

    print('min\t\t\t25%\t\t\tmean\t\t\tstd\t\t\tmedian\t\t\t75%\t\t\tmax')
    print("{}\t\t\t{}\t\t\t{:.2f}\t\t\t{:.2f}\t\t\t{}\t\t\t{}\t\t\t{}".format(np.min(vals), int(np.percentile(vals, 25)), np.mean(vals), np.std(vals), int(np.median(vals)), int(np.percentile(vals, 75)), np.max(vals)))

    consecutive_low_freq = 0
    for f, b1, b2 in zip(norm_freqs, bins[:-1], bins[1:]):
        print('{}-{}:'.format(b1, b2) + '#'*(int(f)))

        if f < 1:
            consecutive_low_freq += 1

        if consecutive_low_freq >= 3:
            break


if __name__ == "__main__":
    connection = utils.create_mysql_connection()
    # Total
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) AS total FROM `10k`"
        cursor.execute(sql)
        results = cursor.fetchall()
    total = results[0]['total']
    print('{} 10k reports in total'.format(total))
    print()

    # If there are some badly parsed files
    print('BADLY PARSED FILES')
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `10k` WHERE {} = %s".format(config.KEY_BADLY_PARSED)
        cursor.execute(sql, (True))
        results = cursor.fetchall()
    print('{} badly parsed files ({:.2f}%)'.format(len(results), 100.0*float(len(results))/total))
    if len(results) > 0:
        print('Few samples:')
        for i in range(0, min(len(results), 5)):
            print('\t' + str(results[i]['file']))
    print()

    print('NO SECTION EXTRACTED')
    with connection.cursor() as cursor:
        attributes = [config.KEY_ITEM_1A_1, config.KEY_ITEM_1B_1, config.KEY_ITEM_2_1, config.KEY_ITEM_7_1, config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1]
        sql = "SELECT * FROM `10k` WHERE " + ' and '.join([a + ' is NULL' for a in attributes])
        cursor.execute(sql)
        results = cursor.fetchall()
    print('There are {} files where no sections have been extracted ({:.2f}%)'.format(len(results), 100.0*float(len(results))/total))
    if len(results) > 0:
        print('Few samples:')
        for i in range(0, min(len(results), 5)):
            print('\t' + ' '.join([k + ':' + str(v) for k, v in results[i].items() if 't_2' not in k and '_3' not in k and '_9' not in k and 'date' not in k and 'badly' not in k and 'cik' not in k and 'year' not in k and 'id' not in k]))
    print()

    print('FILES CONTAINING “1a Risk Factors” SECTION (INDEPENDENTLY OF THE LENGTH)')
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL".format(config.KEY_ITEM_1A_1)
        cursor.execute(sql)
        count = cursor.fetchall()[0]['total']
    print('There are {} files with a “1a Risk Factors” section ({:.2f}%)'.format(count, 100.0*float(count)/total))
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL and ({} is NULL and {} is NULL and {} is NULL)".format(config.KEY_ITEM_1A_1, config.KEY_ITEM_1B_1, config.KEY_ITEM_2_1, config.KEY_ITEM_2_2)
        cursor.execute(sql)
        count2 = cursor.fetchall()[0]['total']
    print('Among which {} are unextractable ({:.2f}%)'.format(count2, 100.0*float(count2)/count))
    print('Distribution of the length')
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `10k` WHERE {} is not NULL and ({} is not NULL or {} is not NULL or {} is not NULL)".format(config.KEY_ITEM_1A_1, config.KEY_ITEM_1B_1, config.KEY_ITEM_2_1, config.KEY_ITEM_2_2)
        cursor.execute(sql)
        vals = []
        wrong_parsing = 0
        for row in cursor:
            if row[config.KEY_ITEM_1B_1] is not None and row[config.KEY_ITEM_1B_1] > row[config.KEY_ITEM_1A_1]:
                vals.append(row[config.KEY_ITEM_1B_1] - row[config.KEY_ITEM_1A_1])
            elif row[config.KEY_ITEM_2_1] is not None and row[config.KEY_ITEM_2_1] > row[config.KEY_ITEM_1A_1]:
                vals.append(row[config.KEY_ITEM_2_1] - row[config.KEY_ITEM_1A_1])
            elif row[config.KEY_ITEM_2_2] is not None and row[config.KEY_ITEM_2_2] > row[config.KEY_ITEM_1A_1]:
                vals.append(row[config.KEY_ITEM_2_2] - row[config.KEY_ITEM_1A_1])
            else:
                if wrong_parsing < 5:
                    print('\t sample badly parsed: ' + ' '.join([k + ':' + str(v) for k, v in row.items() if 't_2' not in k and '_3' not in k and '_9' not in k and 'date' not in k and 'badly' not in k and 'cik' not in k and 'year' not in k and 'id' not in k]))
                    wrong_parsing += 1
        hist(vals, 100)
    print()

    print('FILES CONTAINING "7 Management’s Discussion and Analysis of Financial Condition and Results of Operations” SECTION (INDEPENDENTLY OF THE LENGTH)')
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL".format(config.KEY_ITEM_7_1)
        cursor.execute(sql)
        count = cursor.fetchall()[0]['total']
    print('There are {} files with a "7 Management’s Discussion and Analysis of Financial Condition and Results of Operations” section ({:.2f}%)'.format(count, 100.0*float(count)/total))
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL and ({} is NULL and {} is NULL and {} is NULL)".format(config.KEY_ITEM_7_1, config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1)
        cursor.execute(sql)
        count2 = cursor.fetchall()[0]['total']
    print('Among which {} are unextractable ({:.2f}%)'.format(count2, 100.0*float(count2)/count))
    print('Distribution of the length')
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `10k` WHERE {} is not NULL and ({} is not NULL or {} is not NULL or {} is not NULL)".format(config.KEY_ITEM_7_1, config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1)
        cursor.execute(sql)
        vals = []
        wrong_parsing = 0
        for row in cursor:
            if row[config.KEY_ITEM_7A_1] is not None and row[config.KEY_ITEM_7A_1] > row[config.KEY_ITEM_7_1]:
                vals.append(row[config.KEY_ITEM_7A_1] - row[config.KEY_ITEM_7_1])
            elif row[config.KEY_ITEM_8_1] is not None and row[config.KEY_ITEM_8_1] > row[config.KEY_ITEM_7_1]:
                vals.append(row[config.KEY_ITEM_8_1] - row[config.KEY_ITEM_7_1])
            elif row[config.KEY_ITEM_9_1] is not None and row[config.KEY_ITEM_9_1] > row[config.KEY_ITEM_7_1]:
                vals.append(row[config.KEY_ITEM_9_1] - row[config.KEY_ITEM_7_1])
            else:
                if wrong_parsing < 5:
                    print('\t sample wrong parsing: ' + ' '.join([k + ':' + str(v) for k, v in row.items() if 't_2' not in k and '_3' not in k and '_9' not in k and 'date' not in k and 'badly' not in k and 'cik' not in k and 'year' not in k and 'id' not in k]))
                    wrong_parsing += 1
        hist(vals, 100)
    print()

    print('FILES CONTAINING “7A Quantitative and Qualitative Disclosures about Market Risk” SECTION (INDEPENDENTLY OF THE LENGTH)')
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL".format(config.KEY_ITEM_7A_1)
        cursor.execute(sql)
        count = cursor.fetchall()[0]['total']
    print('There are {} files with a “7A Quantitative and Qualitative Disclosures about Market Risk” section ({:.2f}%)'.format(count, 100.0*float(count)/total))
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) as total FROM `10k` WHERE {} is not NULL and {} is NULL and {} is NULL and {} is NULL".format(config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1, config.KEY_ITEM_9_2)
        cursor.execute(sql)
        count2 = cursor.fetchall()[0]['total']
    print('Among which {} are unextractable ({:.2f}%)'.format(count2, 100.0*float(count2)/count))
    print('Distribution of the length')
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `10k` WHERE {} is not NULL and ({} is not NULL or {} is not NULL or {} is not NULL)".format(config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1, config.KEY_ITEM_9_2)
        cursor.execute(sql)
        vals = []
        wrong_parsing = 0
        for row in cursor:
            if row[config.KEY_ITEM_8_1] is not None and row[config.KEY_ITEM_8_1] > row[config.KEY_ITEM_7A_1]:
                vals.append(row[config.KEY_ITEM_8_1] - row[config.KEY_ITEM_7A_1])
            elif row[config.KEY_ITEM_9_1] is not None and row[config.KEY_ITEM_9_1] > row[config.KEY_ITEM_7A_1]:
                vals.append(row[config.KEY_ITEM_9_1] - row[config.KEY_ITEM_7A_1])
            elif row[config.KEY_ITEM_9_2] is not None and row[config.KEY_ITEM_9_2] > row[config.KEY_ITEM_7A_1]:
                vals.append(row[config.KEY_ITEM_9_2] - row[config.KEY_ITEM_7A_1])
            else:
                if wrong_parsing < 5:
                    print('\t sample wrong parsing: ' + ' '.join([k + ':' + str(v) for k, v in row.items() if 't_2' not in k and '_3' not in k and '_9' not in k and 'date' not in k and 'badly' not in k and 'cik' not in k and 'year' not in k and 'id' not in k]))
                    wrong_parsing += 1
        hist(vals, 100)
    print()

    connection.close()
