import config
from dateutil.parser import parse
from datetime import datetime


# Clean some char in a phrase
def clean_phrase(word):
    return word.replace('_', '').replace(':', '').replace('<', '').replace('>', '').replace('-', ' ').replace('[', '').replace(']', '').replace('.', ' ').replace('day', ' ')


# Better format and Fix error in fiscal year end
def clean_date(fiscal_year_end, year_of_annual_report):
    date = parse(fiscal_year_end)
    day, month, year = date.day, date.month, date.year

    # Case where 1201 has been considered as a year rather than 1st of December
    if year < config.START_YEAR:
        month = fiscal_year_end[:2]
        day = fiscal_year_end[2:4]
        year = year_of_annual_report
    elif year == datetime.now().year: # Year hasn't been found in the pattern and has been 2017 instead
        year = year_of_annual_report

    return "{} {} {}".format(day, month, year) # Non-American format


#============RELEASE DATE================
# Extract release date
def extract_release_date(buffer):
    val = None

    buffer_cleaned = []
    for l in buffer: # To support multi-threading
        buffer_cleaned.append(' '.join(clean_phrase(l).split())) # Also remove multiple-spaces

    # First attempt to extract the fiscal end year
    extracted_release_date = False
    for idx, l in enumerate(buffer_cleaned):
        if not extracted_release_date:
            for kw in config.KEY_WORDS_LOOKUP_RELEASE_DATE_END:
                if kw in l:
                    extracted_release_date, val = _extract_date_util(l, kw)
                    if extracted_release_date:
                        break
            if extracted_release_date:
                break

    # More deep analysis
    if not extracted_release_date:
        for i, l in enumerate(buffer_cleaned):
            l = l.lower()
            for kw in config.KEY_WORDS_LOOKUP_RELEASE_DATE_END:
                if kw in l:
                    extracted_release_date, val = _extract_date_util(l, kw)

                    # If date hasn't been extracted, it means it is in the previous lines or next ones
                    is_second_path = False
                    while not extracted_release_date:
                        for j in range(1, 30):
                            if i - j > 0:
                                extracted_release_date, val = _extract_date_util(buffer_cleaned[i - j].lower(), kw, is_second_path)
                                if extracted_release_date:
                                    break
                            else:
                                break

                            if not extracted_release_date and i + j < (len(buffer_cleaned) - 1):
                                extracted_release_date, val = _extract_date_util(buffer_cleaned[i + j].lower(), kw, is_second_path)
                                if extracted_release_date:
                                    break
                            else:
                                break

                        if not is_second_path:
                            is_second_path = True
                        else:
                            break
                    else:
                        break

            if extracted_release_date:
                break

    return val


#============FISCAL YEAR END================
# Extract raw fiscal end year
def extract_fiscal_end_year(buffer):
    val = None

    buffer_cleaned = []
    for l in buffer: # To support multi-threading
        buffer_cleaned.append(clean_phrase(l))

    # First attempt to extract the fiscal end year
    extracted_fiscal_year = False
    for l in buffer_cleaned:
        if not extracted_fiscal_year:
            for kw in config.KEY_WORDS_LOOKUP_FISCAL_YEAR_END:
                if kw in l:
                    extracted_fiscal_year, val = _extract_date_util(l, kw)
                    if extracted_fiscal_year:
                        break

    # More deep analysis
    if not extracted_fiscal_year:
        for i, l in enumerate(buffer_cleaned):
            l = l.lower()
            for kw in config.KEY_WORDS_LOOKUP_FISCAL_YEAR_END:
                if kw in l:
                    extracted_fiscal_year, val = _extract_date_util(l, kw)

                    # If date hasn't been extracted, it means it is in the previous lines or next ones
                    if not extracted_fiscal_year:
                        for j in range(1, 5):
                            if i - j > 0:
                                extracted_fiscal_year, val = _extract_date_util(buffer_cleaned[i - j].lower(), kw)
                                if extracted_fiscal_year:
                                    break
                            else:
                                break

                            if not extracted_fiscal_year and i + j < (len(buffer_cleaned) - 1):
                                extracted_fiscal_year, val = _extract_date_util(buffer_cleaned[i + j].lower(), kw)
                                if extracted_fiscal_year:
                                    break
                            else:
                                break
                    else:
                        break

                if extracted_fiscal_year:
                    break

    return val


# Try to extract a date given a line
def _extract_date_util(line, kw, split_date=False):
    extracted_succeed = False
    val = None

    if split_date:
        try:
            line = line.split('dated')[1]
            #if 'by' in line:
            #    line = line.split('by')[0]
        except:
            return extracted_succeed, val

    original_extracted_date = None
    try:
        extracted_date = clean_phrase(line.strip().split(kw)[1].strip())
    except:
        # Case where the date is not in the same line
        extracted_date = line.strip()

    original_extracted_date = extracted_date
    empty = False
    # Try to extract the date. If not successful, truncate the end
    while not extracted_succeed and not empty:
        try:
            val = str(parse(extracted_date))
            extracted_succeed = True
        except:
            extracted_date = extracted_date[:-1]
            empty = len(extracted_date) == 0

    empty = False
    # Try to extract the date. If not successful, truncate the beginning
    extracted_date = original_extracted_date
    if not extracted_succeed:
        while not extracted_succeed and not empty:
            try:
                val = str(parse(extracted_date))
                extracted_succeed = True
            except:
                extracted_date = extracted_date[1:]
                empty = len(extracted_date) == 0

    return extracted_succeed, val