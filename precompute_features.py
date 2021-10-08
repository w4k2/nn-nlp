import argparse
import sklearn.model_selection
import numpy as np

import datasets
import lda


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(docs, labels)):
        X_train = [docs[i] for i in train_idx]
        X_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        features_train, features_test = extract_features(X_train, X_test, args.extraction_method)
        np.save(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_X_train.npy', features_train)
        np.save(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_y_train.npy', y_train)
        np.save(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_X_test.npy', features_test)
        np.save(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_y_test.npy', y_test)


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
