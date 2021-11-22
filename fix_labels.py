import datasets
import sklearn.model_selection
from tqdm import tqdm
import numpy as np

def main():
    dataset_name = 'esp_fake'
    attribute = 'title'

    docs, labels = datasets.load_dataset(dataset_name, attribute=attribute)

    acc_all = []

    models = {
        'esp_fake': ('lda', 'tf_idf', 'beto'),
        'bs_detector': ('lda', 'tf_idf'),
        'mixed': ('lda', 'tf_idf', 'beto'),
    }

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        y_train_from_cross_validation, y_test_from_cross_validation = labels[train_idx], labels[test_idx]
        if dataset_name == 'mixed':
            y_train_from_cross_validation[np.argwhere(y_train_from_cross_validation == 2).flatten()] = 0
            y_train_from_cross_validation[np.argwhere(y_train_from_cross_validation == 3).flatten()] = 1
            y_test_from_cross_validation[np.argwhere(y_test_from_cross_validation == 2).flatten()] = 0
            y_test_from_cross_validation[np.argwhere(y_test_from_cross_validation == 3).flatten()] = 1

        filenames_train, y_train = load_labels(dataset_name, attribute, models, fold_idx, 'train')
        # print(y_train.shape)
        # print(y_train)
        # y_train_from_cross_validation = (y_train_from_cross_validation - 1) * -1
        # print(y_train_from_cross_validation.shape)
        # print(y_train_from_cross_validation)
        for filename, y in zip(filenames_train, y_train):
            if not (np.array_equal(y, y_train_from_cross_validation)):
                print(f"labels mismatch for file {filename}")
                if 'beto' in filename: # TODO remove this later
                    continue
                fixed_y = (y - 1) * -1
                print(fixed_y)
                print(y_train_from_cross_validation)
                assert np.array_equal(fixed_y, y_train_from_cross_validation), f'something is realy off for file {filename}'
                np.save(filename, fixed_y)
        filenames_test, y_test = load_labels(dataset_name, attribute, models, fold_idx, 'test')
        for filename, y in zip(filenames_test, y_test):
            print(f"labels mismatch for file {filename}")
            if 'beto' in filename: # TODO remove this later
                continue
            fixed_y = (y - 1) * -1
            print(fixed_y)
            print(y_test_from_cross_validation)
            assert np.array_equal(fixed_y, y_test_from_cross_validation), f'something is realy off for file {filename}'
            np.save(filename, fixed_y)

def load_labels(dataset_name, attribute, models, fold_idx, phase='train'):
    labels_from_file = []
    filenames = []
    for model_name in models[dataset_name]:
        label_filename = f'./extracted_features/{dataset_name}_{model_name}/fold_{fold_idx}_y_{phase}_{attribute}.npy'
        filenames.append(label_filename)
        label = np.load(label_filename)
        labels_from_file.append(label)
    for i in range(1, len(labels_from_file)):
        if models[dataset_name][i] == 'beto': # TODO remove this if after fixing beto
            continue
        if not np.array_equal(labels_from_file[0], labels_from_file[i]): # all labels should be the same
            raise Exception(f'labels for {models[dataset_name][i]} extractions are different than the others!')
    return filenames, labels_from_file

if __name__ == '__main__':
    main()