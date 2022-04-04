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
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
# import sys
# np.set_printoptions(threshold=sys.maxsize) # for table debugging
# python test_concatenated_extractions.py --dataset_name esp_fake --attribute text


def main():
    args = parse_args()
    print(args)

    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)
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

    acc_all = []

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        y_train_from_cross_validation, y_test_from_cross_validation = labels[train_idx], labels[test_idx]

        X_train_all, y_train_all = get_extracted_features_and_labels(args, models, train_datasets, fold_idx, 'train')
        if args.mode == '4M':
            assert(np.array_equal(y_train_all[0], y_train_from_cross_validation))
        X_test_all, y_test_all = get_extracted_features_and_labels(args, models, train_datasets, fold_idx, 'test')
        if args.mode == '4M':
            assert(np.array_equal(y_test_all[0], y_test_from_cross_validation))

        for i in range(len(X_train_all)):
            X_train = X_train_all[i]
            y_train = y_train_all[i]
            X_test = X_test_all[i]
            y_test = y_test_all[i]

            if args.mode == '4M':
                num_features = X_train.shape[1] // 4  # average over number of models
            elif args.mode == '3M':
                num_features = X_train.shape[1] // 3  # average over number of possible training datasets
            elif args.mode == '12M':
                num_features = X_train.shape[1] // 12  # average over both

            selected_X_train, selected_X_test = select_features(args, X_train, y_train, X_test, num_features)

            model = sklearn.neural_network.MLPClassifier((500, 500))
            model.fit(selected_X_train, y_train)

            y_pred = model.predict(selected_X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            acc_all.append(accuracy)
            acc_update = f'fold {fold_idx}, accuracy = {accuracy}' if len(X_train_all) == 1 else f'fold {fold_idx}, model = {models[args.dataset_name][i]}, accuracy = {accuracy}'
            print(acc_update)

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_concat_extraction_model_avrg_{args.feature_selection}_{args.attribute}_{args.mode}.npy', acc_all)


def get_extracted_features_and_labels(args, models, train_datasets, fold_idx, phase='train'):
    if args.mode == '4M':
        loaded_features = []
        loaded_labels = []
        for model_name in models[args.dataset_name]:
            features_filename = f'./extracted_features/{args.dataset_name}_{model_name}_{args.dataset_name}/fold_{fold_idx}_X_{phase}_{args.attribute}.npy'
            features = np.load(features_filename)
            loaded_features.append(features)
            label_filename = f'./extracted_features/{args.dataset_name}_{model_name}_{args.dataset_name}/fold_{fold_idx}_y_{phase}_{args.attribute}.npy'
            label = np.load(label_filename)
            loaded_labels.append(label)
        loaded_features = np.concatenate(loaded_features, axis=1)
        feature_list = [loaded_features]
        for i in range(1, len(loaded_labels)):
            if not np.array_equal(loaded_labels[0], loaded_labels[i]):  # all labels should be the same
                raise Exception(f'labels for {models[args.dataset_name][i]} extractions are different than the others!')
        labels = [loaded_labels[0]]
    elif args.mode == '3M':
        feature_list = []
        labels = []
        for model_name in models[args.dataset_name]:
            model_features = []
            for train_dataset in train_datasets[model_name]:
                features_filename = f'./extracted_features/{train_dataset}_{model_name}_{args.dataset_name}/fold_{fold_idx}_X_{phase}_{args.attribute}.npy'
                features = np.load(features_filename)
                model_features.append(features)
                if train_dataset == args.dataset_name:
                    label_filename = f'./extracted_features/{train_dataset}_{model_name}_{args.dataset_name}/fold_{fold_idx}_y_{phase}_{args.attribute}.npy'
                    label = np.load(label_filename)
                    labels.append(label)
            model_features = np.concatenate(model_features, axis=1)
            feature_list.append(model_features)
            assert model_features.shape[0] == label.shape[0]
        num_models = 5 if args.dataset_name == 'mixed' else 4
        assert len(feature_list) == num_models
        assert len(labels) == num_models
    elif args.mode == '12M':
        feature_list = []
        labels = []
        for model_name in models[args.dataset_name]:
            for train_dataset in train_datasets[model_name]:
                features_filename = f'./extracted_features/{train_dataset}_{model_name}_{args.dataset_name}/fold_{fold_idx}_X_{phase}_{args.attribute}.npy'
                features = np.load(features_filename)
                feature_list.append(features)
                if train_dataset == args.dataset_name:
                    label_filename = f'./extracted_features/{train_dataset}_{model_name}_{args.dataset_name}/fold_{fold_idx}_y_{phase}_{args.attribute}.npy'
                    label = np.load(label_filename)
                    labels.append(label)
        feature_list = [np.concatenate(feature_list, axis=1)]
        for i in range(1, len(labels)):
            if not np.array_equal(labels[0], labels[i]):  # all labels should be the same
                raise Exception(f'labels for {models[args.dataset_name][i]} extractions are different than the others!')
        labels = [labels[0]]

    return feature_list, labels


def select_features(args, X_train, y_train, X_test, num_features):
    if args.feature_selection == 'mutual_info':
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_features)
        feature_selector.fit(X_train, y_train)
        selected_X_train = feature_selector.transform(X_train)
        selected_X_test = feature_selector.transform(X_test)
    elif args.feature_selection == 'anova':
        feature_selector = SelectKBest(score_func=f_classif, k=num_features)
        feature_selector.fit(X_train, y_train)
        selected_X_train = feature_selector.transform(X_train)
        selected_X_test = feature_selector.transform(X_test)
    else:
        num_all_features = min(num_features, len(X_train))
        preliminary_pca = PCA(n_components=num_all_features)
        preliminary_pca.fit(X_train)
        cumulative = np.cumsum(preliminary_pca.explained_variance_ratio_)
        indexes = np.argwhere(cumulative > 0.9)
        num_pca_features = indexes.min() + 1 if len(indexes) > 0 else num_all_features
        pca = PCA(n_components=num_pca_features)
        pca.fit(X_train)
        selected_X_train = pca.transform(X_train)
        selected_X_test = pca.transform(X_test)
    return selected_X_train, selected_X_test


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--feature_selection', type=str, choices=('anova', 'mutual_info', 'pca'))
    parser.add_argument('--mode', type=str, choices=('3M', '4M', '12M'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
