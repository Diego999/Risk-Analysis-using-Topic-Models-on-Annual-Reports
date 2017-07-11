import config
from dateutil.parser import parse
from datetime import datetime


# Clean some char in a phrase
def clean_phrase(word):
    return word.replace('_', '').replace(':', '').replace('<', '').replace('>', '').replace('-', ' ').replace('[', '').replace(']', '')


# Better format and Fix error in fiscal year end
def clean_fiscal_year_end(fiscal_year_end, year_of_annual_report):
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
                    extracted_fiscal_year, val = _extract_final_year_end_util(l, kw)
                    if extracted_fiscal_year:
                        break

    # More deep analysis
    if not extracted_fiscal_year:
        for i, l in enumerate(buffer_cleaned):
            l = l.lower()
            for kw in config.KEY_WORDS_LOOKUP_FISCAL_YEAR_END:
                if kw in l:
                    extracted_fiscal_year, val = _extract_final_year_end_util(l, kw)

                    # If date hasn't been extracted, it means it is in the previous lines or next ones
                    if not extracted_fiscal_year:
                        for j in range(1, 5):
                            if i - j > 0:
                                extracted_fiscal_year, val = _extract_final_year_end_util(buffer_cleaned[i - j].lower(), kw)
                                if extracted_fiscal_year:
                                    break
                            else:
                                break

                            if not extracted_fiscal_year and i + j < (len(buffer_cleaned) - 1):
                                extracted_fiscal_year, val = _extract_final_year_end_util(buffer_cleaned[i + j].lower(), kw)
                                if extracted_fiscal_year:
                                    break
                            else:
                                break
                    else:
                        break

                if extracted_fiscal_year:
                    break

    return val


def _extract_final_year_end_util(line, kw):
    extracted_succeed = False
    val = None

    try:
        extracted_date = clean_phrase(line.strip().split(kw)[1].strip())
    except:
        # Case where the date is not in the same line
        extracted_date = line.strip()

    empty = False
    # Try to extract the date. If not successful, truncate the end
    while not extracted_succeed and not empty:
        try:
            val = str(parse(extracted_date))
            extracted_succeed = True
        except:
            extracted_date = extracted_date[:-1]
            empty = len(extracted_date) == 0

    return extracted_succeed, val