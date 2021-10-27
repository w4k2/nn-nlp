import sklearn.neural_network
import numpy as np
import argparse
import pathlib
import os


def main():
    args = parse_args()

    acc_all = []

    data_path = pathlib.Path(f'extracted_features/{args.dataset_name}_{args.extraction_method}')

    for fold_idx in range(10):
        model = sklearn.neural_network.MLPClassifier((500, 500))

        X_train = np.load(data_path / f'fold_{fold_idx}_X_train.npy')
        y_train = np.load(data_path / f'fold_{fold_idx}_y_train.npy')
        X_test = np.load(data_path / f'fold_{fold_idx}_X_test.npy')
        y_test = np.load(data_path / f'fold_{fold_idx}_y_test.npy')

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

    avrg_acc = sum(acc_all) / len(acc_all)
    print(f'\naverage accuracy = {avrg_acc}')

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_{args.extraction_method}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('nips', 'esp_fake', 'liar', 'bs_detector'))
    parser.add_argument('--extraction_method', type=str, choices=('lda', 'tf_idf'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
