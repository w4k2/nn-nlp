import argparse
from scipy.sparse import data
import sklearn.model_selection
import numpy as np
import os
import pathlib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import datasets
import lda
import tf_idf


def main():
    args = parse_args()

    loaded_datasets = {}
    fold_indexes = {}
    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    for dataset_name in ('esp_fake', 'bs_detector', 'mixed'):
        docs, labels = datasets.load_dataset(dataset_name, attribute=args.attribute)
        loaded_datasets[dataset_name] = docs, labels
        indexes = k_fold.split(docs, labels)
        train_idx, test_idx = zip(*indexes)
        fold_indexes[dataset_name] = train_idx, test_idx

    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)
    pbar = tqdm(range(10), desc='Fold feature extraction', total=10)
    for fold_idx in pbar:
        train_data_idx = fold_indexes[args.dataset_name][0][fold_idx]
        docs_train = [docs[i] for i in train_data_idx]
        model = get_model(args)
        model.fit(docs_train)

        for dataset_name, (train_idx, test_idx) in fold_indexes.items():
            docs, labels = loaded_datasets[dataset_name]
            docs_train = [docs[i] for i in train_idx[fold_idx]]
            docs_test = [docs[i] for i in test_idx[fold_idx]]
            X_train = model.transform(docs_train)
            X_test = model.transform(docs_test)
            if args.extraction_method == 'tf_idf':
                X_train = X_train.toarray()
                X_test = X_test.toarray()
            y_train, y_test = labels[train_idx[fold_idx]], labels[test_idx[fold_idx]]

            assert X_train.shape[0] == y_train.shape[0]
            assert X_test.shape[0] == y_test.shape[0]

            output_path = pathlib.Path(f'extracted_features/{args.dataset_name}_{args.extraction_method}_{dataset_name}')
            os.makedirs(output_path, exist_ok=True)

            for name, array in zip(('X_train', 'y_train', 'X_test', 'y_test'), (X_train, y_train, X_test, y_test)):
                np.save(output_path / f'fold_{fold_idx}_{name}_{args.attribute}.npy', array)


def get_model(args):
    if args.extraction_method == 'lda':
        return lda.LDA(num_features=args.num_features)
    elif args.extraction_method == 'tf_idf':
        return TfidfVectorizer(max_features=args.num_features)
    else:
        raise ValueError('Invalid extraction method')


def extract_features(X_train, X_test, args):
    if args.extraction_method == 'lda':
        return lda.extract_features(X_train, X_test, num_features=args.num_features)
    elif args.extraction_method == 'tf_idf':
        return tf_idf.extract_features(X_train, X_test, num_features=args.num_features)
    else:
        raise ValueError('Invalid extraction method')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--extraction_method', type=str, choices=('lda', 'tf_idf'))
    parser.add_argument('--num_features', type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
