import datasets
import argparse
import os
import sklearn.model_selection
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # do not remove this import, it is required for registring ops
from tqdm import tqdm
# from official.nlp import optimization
from adamw_optimizer import create_optimizer


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    args = parse_args()
    docs, labels = datasets.load_dataset(args.dataset_name)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(docs, labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        docs_train = [docs[i] for i in train_idx]
        docs_test = [docs[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_test = train_model(docs_train, y_train, docs_test, language=args.language)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('nips', 'esp_fake', 'liar',))
    parser.add_argument('--language', type=str, choices=('eng', 'multi'))

    args = parser.parse_args()
    return args


def train_model(x_train, y_train, x_test, language='eng'):
    print(f'\n\n{tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}\n\n')
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
    base_model = train_bert_model(x_train, y_train, model)
    extraction_model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
    train_features = extraction_model.predict(x_train)["pooled_output"]
    test_features = extraction_model.predict(x_test)["pooled_output"]

    return train_features, test_features


def train_bert_model(x_train, y_train, model, epochs=5):
    y_train_c = tf.keras.utils.to_categorical(y_train)
    x_train_c = np.asarray(x_train)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = tf.metrics.CategoricalAccuracy()
    steps_per_epoch = np.sqrt(len(x_train))
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
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
        epochs=epochs
    )

    return model


def get_model(bert_preprocess_url, bert_model_url):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    out = outputs['pooled_output']
    y = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(out)
    model = tf.keras.Model(text_input, y)
    return model


if __name__ == '__main__':
    main()
