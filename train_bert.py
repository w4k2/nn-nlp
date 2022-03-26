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
from sklearn.metrics import balanced_accuracy_score
from utils.adamw_optimizer import create_optimizer


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name, attribute=args.attribute)

    acc_all = []

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [str(docs[i]) for i in train_idx]
        docs_test = [str(docs[i]) for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        num_outputs = 4 if args.dataset_name == 'mixed' else 2
        model = get_model(language=args.language, num_outputs=num_outputs)
        model = train_bert_model(docs_train, y_train, model)
        checkpoint_path = f'./weights/bert_{args.language}/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/bert'
        model.save_weights(checkpoint_path)

        y_pred = model.predict(docs_test)
        accuracy = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_bert_{args.language}_{args.attribute}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--language', type=str, choices=('eng', 'multi'), help='language BERT model was pretrained on')
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


def get_model(language='eng', num_outputs=2):
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
    model = get_keras_model(bert_preprocess_url, bert_model_url, num_outputs=num_outputs)
    return model


def get_keras_model(bert_preprocess_url, bert_model_url, num_outputs=2):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    out = outputs['pooled_output']
    out = tf.keras.layers.Dropout(0.1)(out)
    y = tf.keras.layers.Dense(num_outputs, activation='softmax', name='classifier')(out)
    model = tf.keras.Model(text_input, y)
    return model


def train_bert_model(x_train, y_train, model, init_lr=3e-5, epochs=5):
    y_train_c = tf.keras.utils.to_categorical(y_train)
    x_train_c = np.asarray(x_train)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = tf.metrics.CategoricalAccuracy()
    steps_per_epoch = np.sqrt(len(x_train))
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(
        x=x_train_c,
        y=y_train_c,
        epochs=epochs,
        batch_size=32,
    )

    return model


if __name__ == '__main__':
    # language = 'eng'
    # model = get_model(language=language)
    # checkpoint_path = f'./weights/bert_{language}/fold_{0}/bert'
    # model.save_weights(checkpoint_path)
    # new_model = get_model(language=language)
    # new_model.load_weights(checkpoint_path)
    # for layer, new_layer in zip(model.layers, new_model.layers):
    #     for w, new_w in zip(layer.weights, new_layer.weights):
    #         try:
    #             assert w == new_w
    #         except ValueError:
    #             assert np.allclose(w, new_w)

    main()
