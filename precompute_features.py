import argparse
import sklearn.model_selection
import numpy as np
import os
import pathlib

import datasets
import lda


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name)

    output_path = pathlib.Path(f'extracted_features/{args.dataset_name}_{args.extraction_method}')
    os.makedirs(output_path, exist_ok=True)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(docs, labels)):
        docs_train = [docs[i] for i in train_idx]
        docs_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_test = extract_features(docs_train, docs_test, args.extraction_method)

        for name, array in zip(('X_train', 'y_train', 'X_test', 'y_test'), (X_train, y_train, X_test, y_test)):
            np.save(output_path / f'fold_{fold_idx}_{name}.npy', array)


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
