import argparse
import sklearn.model_selection
import numpy as np
import os
import pathlib
from tqdm import tqdm

import datasets
import lda
import tf_idf


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    output_path = pathlib.Path(f'extracted_features/{args.dataset_name}_{args.extraction_method}')
    os.makedirs(output_path, exist_ok=True)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [docs[i] for i in train_idx]
        docs_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_test = extract_features(docs_train, y_train, docs_test, args)

        for name, array in zip(('X_train', 'y_train', 'X_test', 'y_test'), (X_train, y_train, X_test, y_test)):
            np.save(output_path / f'fold_{fold_idx}_{name}_{args.attribute}.npy', array)


def extract_features(X_train, y_train, X_test, args):
    if args.extraction_method == 'lda':
        return lda.extract_features(X_train, X_test, num_features=args.num_features)
    elif args.extraction_method == 'tf_idf':
        return tf_idf.extract_features(X_train, X_test, num_features=args.num_features)
    else:
        raise ValueError('Invalid extraction method')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--extraction_method', type=str, choices=('lda', 'tf_idf'))
    parser.add_argument('--num_features', type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
