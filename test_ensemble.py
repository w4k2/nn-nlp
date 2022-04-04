import datasets
import argparse
import os
import pathlib
import sklearn.model_selection
import sklearn.neural_network
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


def main():
    args = parse_args()
    print(args)
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    models = {
        'esp_fake': ('bert_multi', 'beto', 'lda', 'tf_idf'),
        'bs_detector': ('bert_eng', 'bert_multi', 'lda', 'tf_idf'),
        'mixed': ('bert_eng', 'bert_multi', 'beto', 'lda', 'tf_idf'),
    }
    train_datasets = {
        'bert_multi': ('esp_fake', 'bs_detector', 'mixed'),
        'beto': ('esp_fake', 'mixed'),
        'bert_eng': ('bs_detector', 'mixed'),
        'lda': ('esp_fake', 'bs_detector', 'mixed'),
        'tf_idf': ('esp_fake', 'bs_detector', 'mixed'),
    }

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        _, y_test = labels[train_idx], labels[test_idx]

        if args.mode == '4M':
            model_predictions = []
            for model_name in models[args.dataset_name]:
                pred_filename = f'./predictions/{model_name}/{args.dataset_name}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
                pred = np.load(pred_filename)
                if args.dataset_name == 'mixed' and pred.shape[1] != 4:
                    pred = np.repeat(pred, 2, axis=1) / 2
                model_predictions.append(pred)
            model_predictions = np.stack(model_predictions)
            average_predictions = np.mean(model_predictions, axis=0, keepdims=False)
            y_pred = np.argmax(average_predictions, axis=1)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            acc_all.append(accuracy)
            print(f'fold {fold_idx}, average models accuracy = {accuracy}')
        elif args.mode == '3M':
            for model_name in models[args.dataset_name]:
                print('model_name = ', model_name)
                model_predictions = []
                for train_dataset in train_datasets[model_name]:
                    pred_filename = f'./predictions/{model_name}/{train_dataset}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
                    pred = np.load(pred_filename)
                    model_predictions.append(pred)
                if not all(pred.shape[1] == 2 for pred in model_predictions):
                    for i in range(len(model_predictions)):
                        if model_predictions[i].shape[1] != 4:
                            model_predictions[i] = np.repeat(model_predictions[i], 2, axis=1) / 2
                model_predictions = np.stack(model_predictions)
                average_predictions = np.mean(model_predictions, axis=0, keepdims=False)
                y_pred = np.argmax(average_predictions, axis=1)
                accuracy = balanced_accuracy_score(y_test, y_pred)
                acc_all.append(accuracy)
                print(f'fold {fold_idx}, model = {model_name}, average models accuracy = {accuracy}')
        elif args.mode == '12M':
            model_predictions = []
            for model_name in models[args.dataset_name]:
                for train_dataset in train_datasets[model_name]:
                    pred_filename = f'./predictions/{model_name}/{train_dataset}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
                    pred = np.load(pred_filename)
                    model_predictions.append(pred)
            if not all(pred.shape[1] == 2 for pred in model_predictions):
                for i in range(len(model_predictions)):
                    if model_predictions[i].shape[1] != 4:
                        model_predictions[i] = np.repeat(model_predictions[i], 2, axis=1) / 2
            model_predictions = np.stack(model_predictions)
            average_predictions = np.mean(model_predictions, axis=0, keepdims=False)
            y_pred = np.argmax(average_predictions, axis=1)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            acc_all.append(accuracy)
            print(f'fold {fold_idx}, average models accuracy = {accuracy}')

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_ensemble_avrg_{args.attribute}_{args.mode}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--mode', type=str, choices=('3M', '4M', '12M'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
