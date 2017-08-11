import os
import utils
from multiprocessing import Process
import shutil
import random
config = __import__('0_config')
process_data = __import__('2a_process_data')


def process_row(row, agg_method, offset_limit, line_to_read_if_no_end_offset, key_interest_1, key_interest_2, key_interest_3, key_next_1, key_next_2, key_next_3=None):
    res = []

    starting_offset = []
    for idx in [key_interest_1, key_interest_2]:
        if row[idx] is not None:
            starting_offset.append(row[idx])
    if row[key_interest_3] is not None:
        starting_offset += [int(v) for v in row[key_interest_3].split(', ')]
    starting_offset = [o for o in starting_offset if o > 0]

    if not (0 < len(starting_offset) <= 3):
        return res
    elif not (len(starting_offset) == 2 and abs(starting_offset[0] - starting_offset[1]) <= offset_limit):
        starting_offset = agg_method(starting_offset)
    else:
        starting_offset = min(starting_offset)

    ending_offset = []
    for idx in [key_next_1, key_next_2, key_next_3]:
        if row[idx] is not None:
            ending_offset.append(row[idx])

    ending_offset = [o for o in ending_offset if o > starting_offset]

    if len(ending_offset) > 0:
        ending_offset = min(ending_offset)
    elif config.READ_SECTION_EVEN_NO_OFFSET_END: # To read text even if we didn't find a clear separation
        ending_offset = starting_offset + line_to_read_if_no_end_offset
    else:
        return res

    # Don't read too much lines
    ending_offset = min(ending_offset, starting_offset + line_to_read_if_no_end_offset)
    assert ending_offset > starting_offset

    res = {'start':starting_offset,
           'end':ending_offset,
           'file':row[config.KEY_FILE],
           'cik':row[config.KEY_CIK],
           'header':'final offsets: {}-{}\t'.format(starting_offset, ending_offset) + ' '.join([k + ':' + str(v) for k, v in sorted(row.items(), key=lambda x: x[0])])}

    return [res]


def fetch_data():
    connection = utils.create_mysql_connection(all_in_mem=False)

    list_1a, list_7, list_7a = [], [], []
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `10k` WHERE {} = %s".format(config.KEY_BADLY_PARSED)
        cursor.execute(sql, (False))

        for row in cursor:
            # 1A risk factors
            list_1a += process_row(row, config.ITEM_1A_AGG_METHOD, config.ITEM_1A_DIFF_OFFSET_KEEP_BOTH, config.ITEM_1A_LINES_TO_READ_IF_NO_NEXT_INDEX, config.KEY_ITEM_1A_1, config.KEY_ITEM_1A_2, config.KEY_ITEM_1A_3, config.KEY_ITEM_1B_1, config.KEY_ITEM_2_1, config.KEY_ITEM_2_2)  # 2_2 because there is section 7 is too far
            # 7 Management's discussion
            list_7 += process_row(row, config.ITEM_7_AGG_METHOD, config.ITEM_7_DIFF_OFFSET_KEEP_BOTH, config.ITEM_7_LINES_TO_READ_IF_NO_NEXT_INDEX, config.KEY_ITEM_7_1, config.KEY_ITEM_7_2, config.KEY_ITEM_7_3, config.KEY_ITEM_7A_1, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1)
            # 7A Quantitative and Qualitative analysis
            list_7a += process_row(row, config.ITEM_7A_AGG_METHOD, config.ITEM_7A_DIFF_OFFSET_KEEP_BOTH, config.ITEM_7A_LINES_TO_READ_IF_NO_NEXT_INDEX, config.KEY_ITEM_7A_1, config.KEY_ITEM_7A_2, config.KEY_ITEM_7A_3, config.KEY_ITEM_8_1, config.KEY_ITEM_9_1, config.KEY_ITEM_9_2)  # 9_2 because there is no a third section afterwards

    connection.close()

    return list_1a, list_7, list_7a


def store_data(elems, folder):
    for elem in elems:
        filename_in = os.path.join(config.DATA_AR_FOLDER, elem['file'].replace(config.EXTENSION_10K_REPORT, config.EXTENSION_CLEAN_PREPROCESSING))
        buffer = []
        with open(filename_in, 'r', encoding='utf-8') as fp:
            for l in fp:
                buffer.append(l)

        buffer = [elem['header'] + '\n'] + buffer[elem['start']:elem['end']]
        filename_out = os.path.join(folder, elem['file'].replace('/', config.CIK_COMPANY_NAME_SEPARATOR))
        with open(filename_out, 'w', encoding='utf-8') as fp:
            for l in buffer:
                fp.write(l)


def work_process(list_1a, list_7, list_7a):
    store_data(list_1a, config.DATA_1A_FOLDER)
    store_data(list_7, config.DATA_7_FOLDER)
    store_data(list_7a, config.DATA_7A_FOLDER)


if __name__ == "__main__":
    random.seed(config.SEED)
    list_1a, list_7, list_7a = fetch_data()

    for folder in [config.DATA_1A_FOLDER, config.DATA_7_FOLDER, config.DATA_7A_FOLDER]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    if config.MULTITHREADING:
        list_1a = utils.chunks(list_1a, 1 + int(len(list_1a) / config.NUM_CORES))
        list_7 = utils.chunks(list_7, 1 + int(len(list_7) / config.NUM_CORES))
        list_7a = utils.chunks(list_7a, 1 + int(len(list_7a) / config.NUM_CORES))
        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=work_process, args=(list_1a[i], list_7[i], list_7a[i])))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        work_process(list_1a, list_7, list_7a)