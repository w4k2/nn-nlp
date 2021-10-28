import smart_open
import tarfile
import re
import numpy as np
import pandas as pd
from sklearn.utils import resample


def load_dataset(dataset_name):
    if dataset_name == 'esp_fake':
        return load_esp_fake()
    elif dataset_name == 'bs_detector':
        return load_bs_detector()
    else:
        raise ValueError('Invalid dataset name')


def load_esp_fake():
    path_to_dataset = "datasets/esp_fake/fakenews-esp.csv"
    file = open(path_to_dataset, 'r')
    text = file.read()
    records = [record.split(",", 1) for record in re.split("\n\d+,", text)][1:]
    labels = np.asarray([convert_to_integer(record[0]) for record in records])
    docs = [validate_string(record[1]) for record in records]
    assert len(labels) == len(docs)
    return docs, labels


def convert_to_integer(label):
    if label in ('True', 'true', 'mostly-true', 'barely-true'):
        return 1
    elif label in ('Fake', 'false', 'half-true', 'pants-fire'):
        return 0
    else:
        raise Exception("Not recognized label in dataset! - %s", label)


def validate_string(doc):
    if(doc == ''):
        raise Exception("None string found during preprocessing!")
    return str(doc)


def load_bs_detector():
    df_words = pd.read_csv('datasets/bs_detector/data.csv')
    y = df_words['label'].values.astype(int)

    base = df_words['text']
    s_idx = np.array(range(len(y))).astype(int)

    resampled = resample(s_idx, n_samples=int(len(y)),
                         replace=False, stratify=y,
                         random_state=42)
    X = base.values.astype('U')
    docs = X[resampled]
    labels = y[resampled]
    return docs, labels
