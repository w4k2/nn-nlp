import datasets
import argparse
import os
import sklearn.model_selection
import sklearn.neural_network
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # do not remove this import, it is required for registring ops
import pathlib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.adamw_optimizer import create_optimizer


class DummyModel():
    def __init__(self, name):
        self.name = name
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.mlp = sklearn.neural_network.MLPClassifier((500, 500)).fit(np.random.rand(300,100), (np.random.rand(300) > 0.5))

    def predict(self, docs):
        tfidf_result = self.tfidf_vectorizer.transform(docs).toarray()
        return self.mlp.predict(tfidf_result)

    def fit(self, docs):
        tfidf_result = self.tfidf_vectorizer.fit(docs)

def load_models(models_list):
    # TODO add real models loading here
    result_models_list = []
    for model in models_list:
        if model == "bert":
            result_models_list.append(DummyModel("bert"))
        elif model == "beto":
            result_models_list.append(DummyModel("beto"))
        elif model == "lda":
            result_models_list.append(DummyModel("lda"))
        elif model == "tfidf":
            result_models_list.append(DummyModel("tfidf"))
    return result_models_list

def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    models = load_models(args.models)
    print(args.models)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [str(docs[i]) for i in train_idx]
        docs_test = [str(docs[i]) for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        if args.dataset_name == 'mixed':
            y_train[np.argwhere(y_train == 2).flatten()] = 0
            y_train[np.argwhere(y_train == 3).flatten()] = 1
            y_test[np.argwhere(y_test == 2).flatten()] = 0
            y_test[np.argwhere(y_test == 3).flatten()] = 1
        predictions = []
        for model in models:
            model.fit(docs_train) #temporary only, to remove after adding real models - classifier must be fit with something
            y_pred = model.predict(docs_test)
            predictions.append(y_pred)

        #averaging predictions
        average_predictions = sum(predictions)/len(models)
        #performing binarization for arguments (probably it will be y_pred.argmax(axis=1) in final solution)
        accuracy = accuracy_score(y_test, average_predictions > 0.5)
        acc_all.append(accuracy)
        print(f'fold {fold_idx}, average models accuracy = {accuracy}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--language', type=str, choices=('eng', 'multi'), help='language BERT model was pretrained on')
    parser.add_argument('--models', type=str, choices=('bert', 'beto', 'lda', 'tfidf'), nargs='+', help='base models for ensemble')
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
