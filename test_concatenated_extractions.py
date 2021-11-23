import datasets
import argparse
import os
import pathlib
import sklearn.model_selection
import sklearn.neural_network
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
# import sys
# np.set_printoptions(threshold=sys.maxsize) # for table debugging
# python test_concatenated_extractions.py --dataset_name esp_fake --attribute text


def get_extracted_features_and_labels(args, models, fold_idx, phase='train'):
        extraction_results = []
        labels_from_file = []
        for model_name in models[args.dataset_name]:
            extraction_result_filename = f'./extracted_features/{args.dataset_name}_{model_name}/fold_{fold_idx}_X_{phase}_{args.attribute}.npy'
            extraction_result = np.load(extraction_result_filename)
            extraction_results.append(extraction_result)
            label_filename = f'./extracted_features/{args.dataset_name}_{model_name}/fold_{fold_idx}_y_{phase}_{args.attribute}.npy'
            label = np.load(label_filename)
            labels_from_file.append(label)
        concatenated_extractions = np.concatenate(extraction_results, axis=1)
        for i in range(1, len(labels_from_file)):
            if not np.array_equal(labels_from_file[0], labels_from_file[i]): # all labels should be the same
                raise Exception(f'labels for {models[args.dataset_name][i]} extractions are different than the others!')
        return concatenated_extractions, labels_from_file[0]

def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    models = {
        'esp_fake': ('bert_multi', 'beto', 'lda', 'tf_idf'),
        'bs_detector': ('bert_eng', 'bert_multi', 'lda', 'tf_idf'),
        'mixed': ('bert_eng', 'bert_multi', 'beto', 'lda', 'tf_idf'),
    }

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        y_train_from_cross_validation, y_test_from_cross_validation = labels[train_idx], labels[test_idx]
        if args.dataset_name == 'mixed':
            y_train_from_cross_validation[np.argwhere(y_train_from_cross_validation == 2).flatten()] = 0
            y_train_from_cross_validation[np.argwhere(y_train_from_cross_validation == 3).flatten()] = 1
            y_test_from_cross_validation[np.argwhere(y_test_from_cross_validation == 2).flatten()] = 0
            y_test_from_cross_validation[np.argwhere(y_test_from_cross_validation == 3).flatten()] = 1

        X_train, y_train = get_extracted_features_and_labels(args, models, fold_idx, 'train')
        assert(np.array_equal(y_train, y_train_from_cross_validation))
        X_test, y_test = get_extracted_features_and_labels(args, models, fold_idx, 'test')
        assert(np.array_equal(y_test, y_test_from_cross_validation))
        number_of_models = len(models[args.dataset_name])
        num_features = X_train.shape[1] // number_of_models
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
            pca = PCA(n_components=min(num_features, len(X_train)))
            pca.fit(X_train)
            selected_X_train = pca.transform(X_train)
            selected_X_test = pca.transform(X_test)

        model = sklearn.neural_network.MLPClassifier((500,500))
        model.fit(selected_X_train, y_train)

        y_pred = model.predict(selected_X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc_all.append(accuracy)
        print(f'fold {fold_idx}, accuracy = {accuracy}')

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_concat_extraction_model_avrg_{args.feature_selection}_{args.attribute}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--feature_selection', type=str, choices=('anova', 'mutual_info', 'pca'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
