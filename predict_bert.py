import datasets
import argparse
import os
import sklearn.model_selection
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # do not remove this import, it is required for registring ops
import pathlib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.adamw_optimizer import create_optimizer


def main():
    args = parse_args()
    predict_docs, predict_labels = datasets.load_dataset(args.prediction_dataset, attribute=args.attribute)
    model_training_docs, model_training_labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)
    k_fold_training = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    training_split = k_fold_training.split(model_training_docs, model_training_labels)
    k_fold_predict = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold_predict.split(predict_docs, predict_labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (_, test_idx) in pbar:
        train_idx, _ = next(training_split)
        docs_train = [str(model_training_docs[i]) for i in train_idx]
        docs_test = [str(predict_docs[i]) for i in test_idx]
        y_train, y_test = model_training_labels[train_idx], predict_labels[test_idx]
        if args.dataset_name == 'mixed':
            y_train[np.argwhere(y_train == 2).flatten()] = 0
            y_train[np.argwhere(y_train == 3).flatten()] = 1
            y_test[np.argwhere(y_test == 2).flatten()] = 0
            y_test[np.argwhere(y_test == 3).flatten()] = 1

        model = get_model(language=args.language)
        checkpoint_path = f'./weights/bert_{args.language}/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/bert'
        model.load_weights(checkpoint_path)

        y_pred = model.predict(docs_test)
        pred_filename = f'./predictions/bert_{args.language}/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
        os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
        np.save(pred_filename, y_pred)

        model_features = get_model(language=args.language, add_classifier=False)
        checkpoint_path = f'./weights/bert_{args.language}/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/bert'
        model_features.load_weights(checkpoint_path)

        output_path = pathlib.Path(f'extracted_features/{args.dataset_name}_bert_{args.language}_{args.prediction_dataset}')
        os.makedirs(output_path, exist_ok=True)
        y_pred = model_features.predict(docs_train)
        np.save(output_path / f'fold_{fold_idx}_X_train_{args.attribute}.npy', y_pred)
        np.save(output_path / f'fold_{fold_idx}_y_train_{args.attribute}.npy', y_train)
        y_pred = model_features.predict(docs_test)
        np.save(output_path / f'fold_{fold_idx}_X_test_{args.attribute}.npy', y_pred)
        np.save(output_path / f'fold_{fold_idx}_y_test_{args.attribute}.npy', y_test)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--language', type=str, choices=('eng', 'multi'), help='language BERT model was pretrained on')
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)
    parser.add_argument('--prediction_dataset', choices=('esp_fake', 'bs_detector', 'mixed'), required=True)

    args = parser.parse_args()
    return args


def get_model(language='eng', add_classifier=True):
    preprocess_url_dict = {
        'eng': "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
        'multi': "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
    }
    model_url_dict = {
        'eng': "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4",
        'multi': "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
    }
    bert_preprocess_url = preprocess_url_dict[language]
    bert_model_url = model_url_dict[language]
    model = get_keras_model(bert_preprocess_url, bert_model_url, add_classifier=add_classifier)
    return model


def get_keras_model(bert_preprocess_url, bert_model_url, add_classifier=True):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    out = outputs['pooled_output']
    if add_classifier:
        out = tf.keras.layers.Dropout(0.1)(out)
        out = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(out)
    model = tf.keras.Model(text_input, out)
    return model

if __name__ == '__main__':
    main()
