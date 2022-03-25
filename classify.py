import sklearn.neural_network
import numpy as np
import argparse
import pathlib
import os
import pickle

from sklearn.metrics import balanced_accuracy_score


def main():
    args = parse_args()
    print(args)

    acc_all = []

    data_path = pathlib.Path(f'extracted_features/{args.train_dataset_name}_{args.extraction_method}_{args.dataset_name}')

    for fold_idx in range(10):
        model = sklearn.neural_network.MLPClassifier((500, 500))

        X_train = np.load(data_path / f'fold_{fold_idx}_X_train_{args.attribute}.npy')
        y_train = np.load(data_path / f'fold_{fold_idx}_y_train_{args.attribute}.npy')
        X_test = np.load(data_path / f'fold_{fold_idx}_X_test_{args.attribute}.npy')
        y_test = np.load(data_path / f'fold_{fold_idx}_y_test_{args.attribute}.npy')

        model.fit(X_train, y_train)

        filename = f'./weights/{args.extraction_method}/{args.train_dataset_name}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/mlp.sav'
        parent_path = os.path.dirname(filename)
        os.makedirs(parent_path, exist_ok=True)
        fp = open(filename, 'wb+')
        pickle.dump(model, fp)

        # accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        accuracy = balanced_accuracy_score(y_test, y_pred)
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

        y_pred = model.predict_proba(X_test)
        # assert y_pred.shape[1] == 2
        pred_filename = f'./predictions/{args.extraction_method}/{args.train_dataset_name}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
        os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
        np.save(pred_filename, y_pred)

    avrg_acc = sum(acc_all) / len(acc_all)
    print(f'\naverage accuracy = {avrg_acc}')

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.train_dataset_name}_{args.extraction_method}_{args.dataset_name}_{args.attribute}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'), help='dataset, that feature extraction method was trained on')
    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--extraction_method', type=str, choices=('lda', 'tf_idf'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
