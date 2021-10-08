import smart_open
import tarfile
import re
import numpy as np


def load_dataset(dataset_name):
    if dataset_name == 'nips':
        return load_nips()
    else:
        raise ValueError('Invalid dataset name')


def load_nips():
    docs = list(extract_nips_documents())
    labels = np.random.randint(low=0, high=2, size=(len(docs)))
    return docs, labels


def extract_nips_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')
