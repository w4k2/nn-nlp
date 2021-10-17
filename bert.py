from gensim.models import LdaModel
from gensim.corpora import Dictionary, dictionary
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import numpy as np


def extract_features(x_train, y_train, x_test, num_features=100):
    base_model = train_bert_model(x_train, y_train, num_topics=num_features)
    extraction_model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
    train_features = extraction_model.predict(x_train)["pooled_output"]
    test_features = extraction_model.predict(x_test)["pooled_output"]

    return train_features, test_features#train_features, test_features


def train_bert_model(x_train, y_train, num_topics=100):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3" #multilanguage
    bert_model_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4" #multilangauge
    # bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" #en_bert
    # bert_model_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1" #en_bert
    preprocessing_layer = hub.KerasLayer(bert_preprocess_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(net)
    model = tf.keras.Model(text_input, net)

    y_train_c = tf.keras.utils.to_categorical(y_train)
    x_train_c = np.asarray(x_train)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = tf.metrics.CategoricalAccuracy()
    epochs = 5
    steps_per_epoch = np.sqrt(len(x_train))
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(
        x = x_train_c,
        y = y_train_c,
        epochs=epochs
    )

    return model
