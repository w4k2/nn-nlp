import smart_open
import tarfile
import re
import numpy as np

def convert_to_integer(label):
    if(label == 'True'):
        return 1
    elif(label == 'Fake'):
        return 0
    else:
        raise Exception("Not recognized label in dataset! - %s", label)

def validate_string(doc):
    if(doc == ''):
        raise Exception("None string found during preprocessing!")
    return str(doc)

def load_dataset(dataset_name):
    if dataset_name == 'nips':
        return load_nips()
    if dataset_name == 'esp_fake':
        return load_esp_fake()
    else:
        raise ValueError('Invalid dataset name')


def load_nips():
    docs = list(extract_nips_documents())
    print(type(docs[0]))
    labels = np.random.randint(low=0, high=2, size=(len(docs)))
    return docs, labels

def load_esp_fake():
    path_to_dataset = "datasets/esp_fake/fakenews-esp.csv"
    file = open(path_to_dataset, 'r')
    text = file.read()
    records = [record.split(",",1) for record in re.split("\n\d+,", text)][1:]
    labels = np.asarray([convert_to_integer(record[0]) for record in records])
    docs = [validate_string(record[1]) for record in records]
    assert len(labels) == len(docs)
    return docs, labels


def extract_nips_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')
