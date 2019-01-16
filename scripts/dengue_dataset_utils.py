import numpy as np
import ntpath
import re

def str_to_indexes(s, length, dict_alphabet):
    s = s.lower()
    max_length = min(len(s), length)
    str2idx = np.zeros(length, dtype='int64')
    for i in range(1, max_length + 1):
        c = s[-i]
        if c in dict_alphabet:
            str2idx[i - 1] = dict_alphabet[c]
    return str2idx


def load_data(data_source):
    data = []
    with open(data_source, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f, delimiter='\t', quotechar='"')
        for row in rdr:
            checker = row[0]
            vals = []
            for val in row[1:]:
                vals.append(val[len(checker) + 1:])

            data.append(tuple(vals))

        return pd.DataFrame(data, columns=names)


def match_all(text, keywords):
    for keyword in keywords:
        matches = re.findall(keyword, text.lower())
        if len(matches) > 0:
            return True
    return False


def treat_data_nn(txt, length, dict_alphabet):
    txt = txt[7:]
    txt = re.sub("\\\/", "/", txt)
    txt = re.sub('@([A-Za-z0-9_]+)', "@", txt)
    txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                 "URL", txt)
    idx = str_to_indexes(txt, length, dict_alphabet)
    return idx


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)