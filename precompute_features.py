import argparse
import sklearn.model_selection

import datasets
import lda


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name)

    k_fold = sklearn.model_selection.RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)
    for train_idx, test_idx in k_fold.split(docs):
        X_train = [docs[i] for i in train_idx]
        X_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        features_train, features_test = extract_features(docs, X_test, args.extraction_method)


def extract_features(X_train, X_test, extraction_method):
    if extraction_method == 'lda':
        return lda.extract_features(X_train, X_test)
    else:
        raise ValueError('Invalid extraction method')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('nips',))
    parser.add_argument('--extraction_method', type=str, choices=('lda',))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
