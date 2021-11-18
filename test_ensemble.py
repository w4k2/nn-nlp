import datasets
import argparse
import os
import sklearn.model_selection
import sklearn.neural_network
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    model_names = ('bert_eng', 'bert_multi', 'beto', 'lda', 'tf_idf')

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        _, y_test = labels[train_idx], labels[test_idx]

        model_predictions = []
        for model_name in model_names:
            pred_filename = f'./predictions/{model_name}/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
            pred = np.load(pred_filename)
            model_predictions.append(pred)
        model_predictions = np.stack(model_predictions)
        average_predictions = np.mean(model_predictions, axis=0, keepdims=False)
        y_pred = np.argmax(average_predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        acc_all.append(accuracy)
        print(f'fold {fold_idx}, average models accuracy = {accuracy}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
