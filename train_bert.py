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
from adamw_optimizer import create_optimizer


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name)
    # for i in range(10):
    #     print(docs[i])
    #     print()
    # exit()

    acc_all = []

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [docs[i] for i in train_idx]
        docs_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        model = train_model(docs_train, y_train, docs_test, language=args.language)
        y_pred = model.predict(docs_test)
        accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_bert_{args.language}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('nips', 'esp_fake', 'liar',))
    parser.add_argument('--language', type=str, choices=('eng', 'multi'))

    args = parser.parse_args()
    return args


def train_model(x_train, y_train, x_test, language='eng'):
    preprocess_url_dict = {
        'eng': "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        'multi': "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
    }
    model_url_dict = {
        'eng': "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
        'multi': "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
    }
    bert_preprocess_url = preprocess_url_dict[language]
    bert_model_url = model_url_dict[language]
    model = get_model(bert_preprocess_url, bert_model_url)
    trained_model = train_bert_model(x_train, y_train, model)
    return trained_model


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


def get_model(bert_preprocess_url, bert_model_url):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    out = outputs['pooled_output']
    out = tf.keras.layers.Dropout(0.1)(out)
    y = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(out)
    model = tf.keras.Model(text_input, y)
    return model


if __name__ == '__main__':
    main()
