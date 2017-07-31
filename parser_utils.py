import config
from dateutil.parser import parse
from datetime import datetime
from bs4 import BeautifulSoup
import re
import distance


# Clean some chars in a phrase
def clean_phrase(phrase):
    return phrase.replace('_', '').replace(':', '').replace('<', '').replace('>', '').replace('-', ' ').replace('[', '').replace(']', '').replace('.', ' ').replace('day', ' ')


def remove_special_chars(phrase):
    return ''.join(c for c in phrase if c.isalnum() or c == ' ')


#Edit distance between two strings
def lev_distance(a, b):
    return distance.levenshtein(remove_special_chars(a).lower(), remove_special_chars(b).lower())


# Better format and Fix error in fiscal year end
def clean_date(fiscal_year_end, year_of_annual_report):
    date = parse(fiscal_year_end)
    day, month, year = date.day, date.month, date.year

    # Case where 1201 has been considered as a year rather than 1st of December
    if year < config.START_YEAR - 1: # Fiscal year may end in 1992 !
        month = fiscal_year_end[:2]
        day = fiscal_year_end[2:4]
        year = year_of_annual_report
    elif year == datetime.now().year: # Year hasn't been found in the pattern and has been 2017 instead
        year = year_of_annual_report

    day = str(day).zfill(2)
    month = str(month).zfill(2)
    year = str(year)
    if len(year) < 4:
        year = ('19' if int(year) > (datetime.now().year-2000) else '20') + year

    try:
        res = datetime.strptime('{}-{}-{}'.format(day, month, year), '%d-%m-%Y')
    except:
        res = None

    return res


#============ITEMS================
def extract_items(buffer):
    vals = None
    try:
        val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = extract_items_util(buffer)

        # Check if there is only 1 occurence starting by "Item XXX" or "Title"
        val_1a = simplify_occurences(val_1a, config.KEY_WORDS_ITEM_1A, buffer)
        val_1b = simplify_occurences(val_1b, config.KEY_WORDS_ITEM_1B, buffer)
        val_2 = simplify_occurences(val_2, config.KEY_WORDS_ITEM_2, buffer)
        val_7 = simplify_occurences(val_7, config.KEY_WORDS_ITEM_7, buffer)
        val_7a = simplify_occurences(val_7a, config.KEY_WORDS_ITEM_7A, buffer)
        val_8 = simplify_occurences(val_8, config.KEY_WORDS_ITEM_8, buffer)
        val_9 = simplify_occurences(val_9, config.KEY_WORDS_ITEM_9, buffer)

        if sum([(1 if len(l) > 1 else 0) for l in
                [val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9]]) != 0:  # Not a unique number per items
            total_length = 0
            while total_length != len(val_1a + val_1b + val_2 + val_7 + val_7a + val_8 + val_9):
                total_length = len(val_1a + val_1b + val_2 + val_7 + val_7a + val_8 + val_9)

                # Remove table of contents refs and before
                val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = remove_toc_and_before(
                    [val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])

                # If there are still several indices, let's try to disambiguate them
                val_1a = disambiguate(val_1a, config.KEY_WORDS_ITEM_1A, buffer)
                val_1b = disambiguate(val_1b, config.KEY_WORDS_ITEM_1B, buffer)
                val_2 = disambiguate(val_2, config.KEY_WORDS_ITEM_2, buffer)
                val_7 = disambiguate(val_7, config.KEY_WORDS_ITEM_7, buffer)
                val_7a = disambiguate(val_7a, config.KEY_WORDS_ITEM_7A, buffer)
                val_8 = disambiguate(val_8, config.KEY_WORDS_ITEM_8, buffer)
                val_9 = disambiguate(val_9, config.KEY_WORDS_ITEM_9, buffer)

                # Remove continuous elements to keep only the first one
                val_1a = remove_continuous(val_1a)
                val_1b = remove_continuous(val_1b)
                val_2 = remove_continuous(val_2)
                val_7 = remove_continuous(val_7)
                val_7a = remove_continuous(val_7a)
                val_8 = remove_continuous(val_8)
                val_9 = remove_continuous(val_9)

        # If there are still more indices per item
        if len(val_1a) > 1 or len(val_1b) > 1 or len(val_2) > 1 or len(val_7) > 1 or len(val_7a) > 1 or len(val_8) > 1 or len(val_9) > 1:
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = remove_first_ref_previous_item([val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])
            val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = remove_last_ref_next_item([val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9])

        vals = [val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9]
    except:
        pass

    return vals


def simplify_occurences(indices, kws, buffer):
    if len(indices) < 1:
        return indices

    buffer_cleaned = []
    for l in buffer:  # To support multi-threading
        buffer_cleaned.append(l.lower())

    final_indices = []
    for idx in indices:
        for kw in kws:
            if len(buffer_cleaned[idx]) >= len(kw) and buffer_cleaned[idx].startswith(kw) and 'continued' not in buffer_cleaned[idx] and 'cont' not in buffer_cleaned[idx] and (('continued' not in buffer_cleaned[idx + 1]) if idx + 1 < len(buffer_cleaned) else True) and buffer_cleaned[idx-1].strip() == '' and '"' not in buffer_cleaned[idx] and '\'' not in buffer_cleaned[idx][-3:] and buffer_cleaned[idx][len(kw)] != '.':
                # Avoid duplicates in header
                different = True
                for i in final_indices:
                    if buffer[i] == buffer[idx]:
                        different = False
                        break
                if different:
                    final_indices.append(idx)
                    break

    return final_indices


def disambiguate(indices, kws, buffer):
    if len(indices) <= 1:
        return indices

    buffer_cleaned = []
    for l in buffer:  # To support multi-threading
        buffer_cleaned.append(l.lower())

    # Count #kw found in the phrase
    indices_matches = []
    for idx in indices:
        counter = 0
        for kw in kws:
            if remove_special_chars(kw) in remove_special_chars(buffer_cleaned[idx]):
                counter += 1
        indices_matches.append(counter)
    max_val = max(indices_matches)

    if indices_matches.count(max_val) == 1:
        return [indices[indices_matches.index(max_val)]]
    elif len(indices) == 2 and indices[0] < 500: # Maybe there is also the TOC
        return [indices[-1]]

    # Compute levenstein distance
    distances = []
    for idx in indices:
        local_distances = []
        for kw in kws:
            local_distances.append(lev_distance(buffer_cleaned[idx], kw))
        distances.append(min(local_distances))
    min_dist = min(distances)

    if distances.count(min_dist) == 1:
        return [indices[distances.index(min_dist)]]
    else:
        final_indices = []
        for idx, d in enumerate(distances):
            if d == min_dist:
                final_indices.append(indices[idx])
        return final_indices


def remove_above(v, _list):
    return [x for x in _list if x < v]


def remove_belove(v, _list):
    return [x for x in _list if x > v]


# [246, 256, 412] [252, 312] -> [246, 256] [252, 312]
def remove_last_ref_next_item(lists):
    elems = [x for x in reversed(range(0, len(lists)))]
    for idx1, i in enumerate(elems):
        if len(lists[i]) > 0:
            max_ref = max(lists[i])
            for j in range(idx1 + 1, len(elems)):
                lists[elems[j]] = remove_above(max_ref, lists[elems[j]])
    return lists


# [246, 256, 412] [216, 312] -> [246, 256, 412] [312]
def remove_first_ref_previous_item(lists):
    for idx1, list in enumerate(lists):
        if len(list) > 0:
            min_ref = min(list)
            for j in range(idx1 + 1, len(lists)):
                lists[j] = remove_belove(min_ref, lists[j])
    return lists


def remove_continuous(_list, max_diff=5):
    if len(_list) <= 1:
        return _list

    new_list = [_list[0]]
    for i in range(1, len(_list)):
        if _list[i] - _list[i-1] > max_diff:
            new_list.append(_list[i])

    return new_list


def remove_toc_and_before(lists, max_diff=2): # Values higher than 2 might cause overlap with other sections
    if sum([len(l) for l in lists]) == 0:
        return lists

    new_lists = []
    temp = []
    mins = []
    for list in lists:
        if len(list) > 1:
            mins.append(min(list))
        if len(list) > 0:
            temp = temp + list
    if len(mins) <= 1:
        return lists
    max_min = max(mins)

    temp.sort()

    # All sections are present in TOC
    if len(mins) == sum([(1 if len(l) > 1 else 0) for l in lists]):
        to_remove = set(mins)
    else:
        to_remove = {temp[0]}

    for i in range(1, len(temp)):
        if temp[i] - max_min <= max_diff: # Not going to far, only table of content
            if temp[i] - temp[i-1] <= max_diff:
                to_remove.add(temp[i])
                to_remove.add(temp[i-1])

    # TOC has at least 2 elements in the section
    if len(to_remove) > 1:
        for list in lists:
            for x in to_remove:
                list = [v for v in list if v > x]
            new_lists.append(list)
    else:
        new_lists = lists

    return new_lists


def find_kw_in_line(kws, idx, line, container):
    for kw in kws:
        if kw in line:
            container.append(idx)
            return True
    return False


def extract_items_util(buffer):
    val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9 = [], [], [], [], [], [], []
    buffer_cleaned = []
    for l in buffer:  # To support multi-threading
        buffer_cleaned.append(l.lower())

    for idx, l in enumerate(buffer_cleaned):
        if find_kw_in_line(config.KEY_WORDS_ITEM_1A, idx, l, val_1a):
            continue
        if find_kw_in_line(config.KEY_WORDS_ITEM_1B, idx, l, val_1b):
            continue
        if find_kw_in_line(config.KEY_WORDS_ITEM_2, idx, l, val_2):
            continue
        # First Item 7A because Item 7 € Item 7a
        if find_kw_in_line(config.KEY_WORDS_ITEM_7A, idx, l, val_7a):
            continue
        if find_kw_in_line(config.KEY_WORDS_ITEM_7, idx, l, val_7):
            continue
        if find_kw_in_line(config.KEY_WORDS_ITEM_8, idx, l, val_8):
            continue
        if find_kw_in_line(config.KEY_WORDS_ITEM_9, idx, l, val_9):
            continue

    return val_1a, val_1b, val_2, val_7, val_7a, val_8, val_9


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


#============GLOBAL CLEANING================
pattern_trim = re.compile(r'\s+')
pattern_html_tags = re.compile('<.*?>')


def remove_special_chars(string):
    return string.replace(',', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('*', '')


def is_number(string):
    try:
        float(remove_special_chars(string))
        return True
    except:
        return False


def is_date(string):
    try:
        parse(remove_special_chars(string))
        return True
    except:
        return False


def check_empty_lines(buffer, idx, before=True, nb_lines=config.MAX_EMPTY_LINES):
    coeff = -1 if before else 1
    for j in range(1, nb_lines + 1):
        if buffer[idx + coeff*j] != '':
            return False

    return True


def clean_raw_text(buffer):
    text = ''.join(buffer)
    b = BeautifulSoup(text, "lxml")
    text = b.get_text().split('\n')

    return text


def parse_text(raw_text):
    buffer = []
    nb_empty_lines = 0
    for l in raw_text:
        l = re.sub(pattern_trim, ' ', l.strip())
        l_lower = l.lower()

        if l == '' or re.search(pattern_html_tags, l) is not None or l in ['$', '%', '*', '#'] or is_number(l) or is_date(l) or len(l) <= config.MIN_LENGTH_LINE or l_lower in ['part i', 'part ii', 'part iii', 'part iv', 'description', 'table of contents', 'exhibit no.']:
            l = ''
            nb_empty_lines += 1
        else:
            nb_empty_lines = 0

        if nb_empty_lines <= config.MAX_EMPTY_LINES:
            buffer.append(l)

    return buffer


def remove_alone_sentence(buffer):
    buffer_temp = []
    for i in range(config.MAX_EMPTY_LINES, len(buffer) - config.MAX_EMPTY_LINES):
        previous = check_empty_lines(buffer, i, before=True, nb_lines=config.MAX_EMPTY_LINES)
        after = check_empty_lines(buffer, i, before=False, nb_lines=config.MAX_EMPTY_LINES)

        if not previous or not after:
            buffer_temp.append(buffer[i])

    return buffer_temp


def remove_excess_empty_lines(buffer, nb_lines=config.MAX_EMPTY_LINES):
    # Remove beginning and end empty lines
    while buffer[0] == '':
        buffer = buffer[1:]
    while buffer[-1] == '':
        buffer = buffer[:-1]

    buffer_final = []
    for i in range(0, len(buffer) - nb_lines):
        if not check_empty_lines(buffer, i - 1, before=False, nb_lines=nb_lines + 1):
            buffer_final.append(buffer[i])

    return buffer_final


def clean_file(buffer):
    try:
        raw_text = clean_raw_text(buffer)
        buffer = parse_text(raw_text)
        buffer = remove_alone_sentence(buffer)
        buffer = remove_excess_empty_lines(buffer)
        return buffer
    except:
        return []