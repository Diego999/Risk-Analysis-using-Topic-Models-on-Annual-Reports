import os
import glob
import re
config = __import__('0_config')


def read_section(section):
    if not os.path.isdir(section):
        print('ERR: {} does not exists !'.format(section))
        exit(-1)

    return glob.glob(section + '/*.' + config.EXTENSION_10K_REPORT)


def remove_header_and_multiple_lines(buffer):
    text = ''.join(buffer[1:])
    text = re.sub(r'\n+', '\n', text).strip()
    return text.split('\n')


def content_relevant(buffer):
    keywords = ['smaller reporting company', 'pages', 'discussion and analysis of financial condition']
    text = ''.join(buffer).lower()

    if len(text) < 400:
        return False

    if len(buffer) < 50 and len(text) < 5000:
        for kw in keywords:
            if kw in text:
                return False

    return True


pattern_remove_multiple_space = re.compile(r'\s+')
pattern_remove_new_lines = re.compile(r'\n+')
pattern_non_char_digit_hash_perc = re.compile(r'[^a-z0-9$% ]+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
def clean(buffer):
    text = ''.join(buffer).lower()
    text = re.sub(pattern_html_tags, '', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_non_char_digit_hash_perc, '', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    return text.strip()


def load_and_clean_data(section):
    final_file = section + '.' + config.EXTENSION_10K_REPORT
    filtered_items = []
    if not os.path.exists(final_file):
        items = read_section(section)

        for item in items:
            buffer = []
            with open(item, 'r', encoding='utf-8') as fp:
                for l in fp:
                    buffer.append(l)

            buffer = remove_header_and_multiple_lines(buffer)
            if content_relevant(buffer):
                filtered_items.append((item, clean(buffer)))

        print('{}: Keep {} over {} ({:.2f}%)'.format(section[section.rfind('/') + 1:], len(filtered_items), len(items), float(len(filtered_items)) / len(items) * 100.0))

        with open(final_file, 'w', encoding='utf-8') as fp:
            for i, l in filtered_items:
                fp.write(i + '\t' + l + '\n')
    else:
        with open(final_file, 'r', encoding='utf-8') as fp:
            for i, l in fp:
                filtered_items.append((i, l.strip()))

    return filtered_items


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]

    for section in sections_to_analyze:
        data = load_and_clean_data(section)
