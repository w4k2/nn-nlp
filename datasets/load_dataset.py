import smart_open
import tarfile
import re
import numpy as np
import pandas as pd


def load_dataset(dataset_name):
    if dataset_name == 'nips':
        return load_nips()
    if dataset_name == 'esp_fake':
        return load_esp_fake()
    if dataset_name == 'liar':
        return load_liar()
    else:
        raise ValueError('Invalid dataset name')


def load_nips():
    docs = list(extract_nips_documents())
    print(type(docs[0]))
    labels = np.random.randint(low=0, high=2, size=(len(docs)))
    return docs, labels


def extract_nips_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')


def load_esp_fake():
    path_to_dataset = "datasets/esp_fake/fakenews-esp.csv"
    file = open(path_to_dataset, 'r')
    text = file.read()
    records = [record.split(",", 1) for record in re.split("\n\d+,", text)][1:]
    labels = np.asarray([convert_to_integer(record[0]) for record in records])
    docs = [validate_string(record[1]) for record in records]
    assert len(labels) == len(docs)
    return docs, labels


def load_liar():
    path_to_dataset = "datasets/liar_dataset/train.tsv"
    df = read_dataframe(path_to_dataset)
    dataset = df.to_numpy()
    labels = np.asarray([convert_to_integer(record[1]) for record in dataset])
    docs = [record[2] + " " + record[3] + " " + record[4]
            for record in dataset]

    return docs, labels


def read_dataframe(tsv_file: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_file, delimiter='\t', dtype=object)
    df.fillna("", inplace=True)
    df.columns = [
        'id',                # Column 1: the ID of the statement ([ID].json).
        'label',             # Column 2: the label.
        'statement',         # Column 3: the statement.
        'subjects',          # Column 4: the subject(s).
        'speaker',           # Column 5: the speaker.
        'speaker_job_title',  # Column 6: the speaker's job title.
        'state_info',        # Column 7: the state info.
        'party_affiliation',  # Column 8: the party affiliation.
        'count_1',  # barely true counts.
        'count_2',  # false counts.
        'count_3',  # half true counts.
        'count_4',  # mostly true counts.
        'count_5',  # pants on fire counts.
        'context'
    ]
    return df


def convert_to_integer(label):
    if(label == 'True'):
        return 1
    elif(label == 'Fake'):
        return 0
    elif(label == 'true'):
        return 1
    elif(label == 'false'):
        return 0
    elif(label == 'half-true'):
        return 0
    elif(label == 'pants-fire'):
        return 0
    elif(label == 'mostly-true'):
        return 1
    elif(label == 'barely-true'):
        return 1
    else:
        raise Exception("Not recognized label in dataset! - %s", label)


def validate_string(doc):
    if(doc == ''):
        raise Exception("None string found during preprocessing!")
    return str(doc)
