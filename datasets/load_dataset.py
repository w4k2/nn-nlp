import smart_open
import tarfile
import re
import numpy as np
import pandas as pd
from sklearn.utils import resample


def load_dataset(dataset_name, attribute='text'):
    if dataset_name == 'esp_fake':
        return load_esp_fake(attribute=attribute)
    elif dataset_name == 'bs_detector':
        return load_bs_detector(attribute=attribute)
    else:
        raise ValueError('Invalid dataset name')


def load_esp_fake(attribute='text'):
    df_attribute = 'Text' if attribute == 'text' else 'Headline'
    df = pd.read_csv('datasets/esp_fake/fakenews-esp.csv')
    docs = [validate_string(d) for d in df[df_attribute]]
    labels = np.asarray([convert_to_integer(l) for l in df['Category']])
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


def load_bs_detector(attribute='text'):
    df_words = pd.read_csv('datasets/bs_detector/data.csv')
    base = df_words[attribute]
    X = base.values.astype('U')

    y = df_words['label'].values.astype(int)
    s_idx = np.array(range(len(y))).astype(int)
    resampled = resample(s_idx, n_samples=int(len(y)),
                         replace=False, stratify=y,
                         random_state=42)
    docs = X[resampled]
    labels = y[resampled]
    return docs, labels
