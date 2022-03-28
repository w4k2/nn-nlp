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
    tf.get_logger().setLevel('ERROR')  # tensorflow prints a lot of warnings due to lack of optimizer in this script
    args = parse_args()
    print(args)
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [str(docs[i]) for i in train_idx]
        docs_test = [str(docs[i]) for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = get_model(language=args.language)
        checkpoint_path = f'./weights/bert_{args.language}/{args.train_dataset_name}/{args.attribute}/fold_{fold_idx}/bert'
        model.load_weights(checkpoint_path)

        y_pred = model.predict(docs_test)
        pred_filename = f'./predictions/bert_{args.language}/{args.train_dataset_name}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
        os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
        np.save(pred_filename, y_pred)

        model_features = get_model(language=args.language, add_classifier=False)
        checkpoint_path = f'./weights/bert_{args.language}/{args.train_dataset_name}/{args.attribute}/fold_{fold_idx}/bert'
        model_features.load_weights(checkpoint_path)

        output_path = pathlib.Path(f'extracted_features/{args.train_dataset_name}_bert_{args.language}_{args.dataset_name}')
        os.makedirs(output_path, exist_ok=True)
        y_pred = model_features.predict(docs_train)
        np.save(output_path / f'fold_{fold_idx}_X_train_{args.attribute}.npy', y_pred)
        np.save(output_path / f'fold_{fold_idx}_y_train_{args.attribute}.npy', y_train)
        y_pred = model_features.predict(docs_test)
        np.save(output_path / f'fold_{fold_idx}_X_test_{args.attribute}.npy', y_pred)
        np.save(output_path / f'fold_{fold_idx}_y_test_{args.attribute}.npy', y_test)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_name', choices=('esp_fake', 'bs_detector', 'mixed'), required=True, help='dataset model was trained on')
    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'), help='currently processed dataset')
    parser.add_argument('--language', type=str, choices=('eng', 'multi'), help='language BERT model was pretrained on')
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

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
