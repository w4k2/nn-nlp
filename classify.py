import sklearn.neural_network
import numpy as np
import argparse


def main():
    args = parse_args()

    acc_all = []

    for fold_idx in range(10):
        model = sklearn.neural_network.MLPClassifier()

        X_train = np.load(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_X_train.npy')
        y_train = np.load(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_y_train.npy')
        X_test = np.load(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_X_test.npy')
        y_test = np.load(f'{args.dataset_name}_fold_{fold_idx}_{args.extraction_method}_y_test.npy')

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

    avrg_acc = sum(acc_all) / len(acc_all)
    print(f'\naverage accuracy = {avrg_acc}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('nips',))
    parser.add_argument('--extraction_method', type=str, choices=('lda',))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
