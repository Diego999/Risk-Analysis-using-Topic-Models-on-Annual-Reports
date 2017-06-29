import os
import datetime
import glob
import urllib.request
import tqdm
import gzip
import pandas as pd
import re


def clean_row(row):
    return row.decode('utf-8', 'ignore').strip()


def compute_year_quarter(START_YEAR, START_QUARTER):
    today = datetime.datetime.now()
    current_year = today.year
    current_quarter = (today.month-1)//3 + 1

    years = list(range(START_YEAR, current_year + 1))
    quarters = list(range(1,4 + 1))
    year_x_quarter = [(y, q) for y in years for q in quarters]
    # We didn't reach the fourth quarter, have to filter last quarters
    if current_quarter < 4:
        year_x_quarter = year_x_quarter[0:-(4-current_quarter)]

    # The first report is not among the first quarter
    if START_QUARTER > 1:
        year_x_quarter = year_x_quarter[START_QUARTER-1:
        ]
    return year_x_quarter


def download_files(year_x_quarter, URL_INDEX_PATTERN, DATA_GZ_FOLDER, START_YEAR):
    if not os.path.isdir(DATA_GZ_FOLDER):
        os.makedirs(DATA_GZ_FOLDER)

    # Don't download already downloaded files
    for id, gz_file in enumerate(glob.glob(DATA_GZ_FOLDER + '/*.gz')):
        y, q = [int(x) for x in gz_file[gz_file.rfind('/') + 1 : gz_file.rfind('.')].split('_')]
        idx = (y - START_YEAR) * 4 + q - 1
        idx -= id # Removing an element in the list will translate all indices

        assert (y,q) == year_x_quarter[idx]

        del year_x_quarter[idx]

    # Download GZ files
    for y, q in tqdm.tqdm(year_x_quarter, desc='Downloading company indices'):
        url = URL_INDEX_PATTERN.format(quarter=q, year=y)
        filename = os.path.join(DATA_GZ_FOLDER, '{year}_{quarter}.gz'.format(year=y, quarter=q))

        urllib.request.urlretrieve(url, filename)


def read_data(DATA_GZ_FOLDER, DATA_PD_FOLDER, PDF_MERGE_FILE):
    extra_keys = ['year', 'quarter']

    if not os.path.isdir(DATA_PD_FOLDER):
        os.makedirs(DATA_PD_FOLDER)

    pdfs = {}
    keys = None
    pattern = re.compile("   *")
    can_safely_load_pdf_merge = True
    for file in tqdm.tqdm(glob.glob(DATA_GZ_FOLDER + '/*.gz'), desc='Processing company indices'):
        y, q = [int(x) for x in file[file.rfind('/')+1:file.rfind('.')].split('_')]

        if y not in pdfs:
            pdfs[y] = {}
        if q not in pdfs[y]:
            pdfs[y][q] = []

        filename = os.path.join(DATA_PD_FOLDER, '{year}_{quarter}.pd'.format(year=y, quarter=q))

        # Read dataframe or process GZ file it if not exists
        if os.path.isfile(filename):
            pdfs[y][q] = pd.read_pickle(filename)
            if keys is None:
                keys = list(pdfs[y][q].columns.values)
        else:
            can_safely_load_pdf_merge = False
            # Read file
            with gzip.open(file, 'r') as fp:
                for i in range(8):
                    next(fp)
                if keys is None:
                    keys = re.split(pattern, clean_row(fp.readline())) + extra_keys
                else:
                    next(fp)
                next(fp)

                # Raw data
                for row in fp:
                    row = clean_row(row)

                    attributes = []
                    attribute = ''
                    spaces = 0
                    for c in row[::-1]:
                        if c == ' ':
                            spaces += 1
                        else:
                            spaces = 0

                        if spaces < 2:
                            attribute += c
                        elif attribute != ' ':
                            attributes.append(attribute[::-1].strip())
                            attribute = ''
                            spaces = 0

                    if attribute != '':
                        attributes.append(attribute[::-1].strip())

                    attributes = attributes[::-1]

                    if len(attributes) >= (len(keys) - len(extra_keys)):
                        while len(attributes) > (len(keys) - len(extra_keys)):
                            attributes[0] += ' ' + attributes[1]
                            del attributes[1]
                        pdfs[y][q].append(attributes)

            # Transform to Pandas dataframe
            pdfs[y][q] = pd.DataFrame.from_records([t + [y, q] for t in pdfs[y][q]], columns=keys)
            pdfs[y][q].to_pickle(os.path.join(DATA_PD_FOLDER, '{year}_{quarter}.pd'.format(year=y, quarter=q)))

    pdfs_merged = None
    if not can_safely_load_pdf_merge:
        pdfs_merged = pd.DataFrame([], columns=keys)
        for pdfs in tqdm.tqdm([pdfs[y][q] for y in sorted(pdfs.keys()) for q in sorted(pdfs[y].keys())], desc='Combining Pandas DataFrames'):
            pdfs_merged = pdfs_merged.append(pdfs, ignore_index=True)
        pdfs_merged.to_pickle(PDF_MERGE_FILE)
    else:
        pdfs_merged = pd.read_pickle(PDF_MERGE_FILE)

    return pdfs, pdfs_merged, keys

if __name__ == "__main__":
    DATA_FOLDER = os.path.join('.', 'data/')
    DATA_GZ_FOLDER = os.path.join(DATA_FOLDER, 'gz')
    DATA_PD_FOLDER = os.path.join(DATA_FOLDER, 'pd')
    DATA_COMPANY_FOLDER = os.path.join(DATA_FOLDER, 'company')
    URL_INDEX_PATTERN = 'https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.gz'
    PDF_MERGE_FILE = os.path.join(DATA_PD_FOLDER, 'merged_pds.pd')

    START_YEAR = 1993
    START_QUARTER = 1

    year_x_quarter = compute_year_quarter(START_YEAR, START_QUARTER)

    download_files(year_x_quarter, URL_INDEX_PATTERN, DATA_GZ_FOLDER, START_YEAR)

    pdfs, pdfs_merge, keys = read_data(DATA_GZ_FOLDER, DATA_PD_FOLDER, PDF_MERGE_FILE)

    print(pdfs_merge.head(5))