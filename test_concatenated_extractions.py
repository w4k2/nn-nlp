import datasets
import argparse
import os
import pathlib
import sklearn.model_selection
import sklearn.neural_network
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# import sys
# np.set_printoptions(threshold=sys.maxsize) # for table debugging
# python test_concatenated_extractions.py --dataset_name esp_fake --attribute text

def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    models = {
        'esp_fake': ('lda', 'tf_idf', 'beto'),
        'bs_detector': ('lda', 'tf_idf'),
        'mixed': ('lda', 'tf_idf', 'beto'),
    }

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        y_train, y_test = labels[train_idx], labels[test_idx]
        if args.dataset_name == 'mixed':
            y_train[np.argwhere(y_test == 2).flatten()] = 0
            y_train[np.argwhere(y_test == 3).flatten()] = 1
            y_test[np.argwhere(y_test == 2).flatten()] = 0
            y_test[np.argwhere(y_test == 3).flatten()] = 1

        extraction_results = []
        labels_from_file = []
        for model_name in models[args.dataset_name]:
            extraction_result_filename = f'./extracted_features/{args.dataset_name}_{model_name}/fold_{fold_idx}_X_train_{args.attribute}.npy'
            extraction_result = np.load(extraction_result_filename)
            extraction_results.append(extraction_result)
            label_filename = f'./extracted_features/{args.dataset_name}_{model_name}/fold_{fold_idx}_y_train_{args.attribute}.npy'
            label = np.load(label_filename)
            labels_from_file.append(label)
        
        concatenated_extractions = np.concatenate(extraction_results, axis=1)
        assert(np.array_equal(labels_from_file[0], labels_from_file[1])) # LDA and TF_IDF IS THE SAME
        assert(np.array_equal(labels_from_file[0], labels_from_file[2])) # BETO is different
        #ssert(np.array_equal(labels_from_file[0], y_train)) #WHY IT DOES NOT WORK?
        print("concatenated_extractions shape: ", concatenated_extractions.shape)
        print("labels shape: ", labels_from_file[0].shape)
        

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_concat_extraction_model_avrg_{args.attribute}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
