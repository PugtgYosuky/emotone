import os
import json
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LogisticRegression

import midi_encoder
from train_generative import build_generative_model

import plot_results

SAVE_CHECKPOINTS = os.path.join('trained')


def preprocess_sentence(text, front_pad='\n', end_pad=''):
    text = text.replace('\n', ' ').strip()
    text = front_pad + text + end_pad
    return text


def encode_sentence(model, text, vocabulary, layer_index):
    text = preprocess_sentence(text)

    # reset hidden layers
    model.reset_states()

    for char in text.split(' '):
        # add batch dimension

        try:
            input_val = tf.expand_dims([vocabulary[char]], 0)
            predictions = model(input_val)
        except KeyError:
            if char != '':
                print('Cannot process char', char)
    h, c = model.get_layer(index=layer_index).states

    c = tf.squeeze(c, 0)
    # h = tf.squeeze(h,0)

    return tf.math.tanh(c).numpy()


def build_dataset(path, generative_model, vocabulary, layer_index):
    x = []
    y = []

    data = pd.read_csv(path)

    for row_index in range(data.shape[0]):
        label = data.iloc[[row_index], ['emotion']]
        filename = data.iloc[[row_index], ['midi']]
        music_name = filename.split('.')[0]
        phrase_path = os.path.join(path, filename)
        encoded_path = os.path.join(path, music_name + '.npt')

        # load midi as text
        if os.path.isfile(encoded_path):
            encoding = np.load(encoded_path)
        else:
            text, _ = midi_encoder.load(phrase_path, transpose_range=1, stretching_range=1)

            # encode with lstm
            encoding = encode_sentence(generative_model, text, vocabulary, layer_index)

            # save encoding
            np.save(encoded_path, encoding)
        x.append(encoding)
        y.append(label)

    return np.array(x), np.array(y)


def get_activated_neurons(sentiment_classifier):
    neurons_not_zero = len(np.argwhere(sentiment_classifier.coef_))

    weights = sentiment_classifier.coef_.T
    weights_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if neurons_not_zero == 1:
        neurons_indexes = np.array([np.argmax(weights_penalties)])
    elif neurons_not_zero >= np.log(len(weights_penalties)):
        neurons_indexes = np.argsort(weights_penalties)[-neurons_not_zero:][::-1]
    else:
        neurons_indexes = np.argpartition(weights_penalties, -neurons_not_zero)[-neurons_not_zero]
        neurons_indexes = (neurons_indexes[np.argsort(weights_penalties[neurons_indexes])])[::-1]

    return neurons_indexes


def train_classifier_model(train_dataset, test_dataset, reg_strength=2 ** np.arange(-8, 1).astype(np.float), seed=42,
                           penalty='l1'):
    train_x, train_y = train_dataset
    test_x, test_y = test_dataset

    scores = []

    # hyperparameter optimization
    for i, j in enumerate(reg_strength):
        logistisc_model = LogisticRegression(C=j, penalty=penalty, random_state=seed + i, solver='liblinear')
        logistisc_model.fit(train_x, train_y)
        score = logistisc_model.score(test_x, test_y)
        scores.append(score)

    # calculate best model
    best_reg_strength = reg_strength(np.argmax(scores))
    final_classifier = LogisticRegression(C=best_reg_strength, penalty=penalty, random_state=seed + len(reg_strength),
                                          solver='liblinear')
    final_classifier.fit(train_x, train_y)
    score = final_classifier.score(test_x, test_y) * 100

    # save model
    with open(os.path.join(SAVE_CHECKPOINTS, 'sentiment_classifier.p'), 'wb') as file:
        pickle.dump(final_classifier, file)

    # get activated neurons

    sentiment_neurons = get_activated_neurons(final_classifier)

    # plot results
    plot_results.plot_weight_contributions(final_classifier.coef_)
    plot_results.plot_logistcs(train_x, train_y, sentiment_neurons)

    return sentiment_neurons, score


def main():
    # variables
    vocabulary_path = os.path.join(SAVE_CHECKPOINTS, "vocabulary_dict.json")
    model_checkpoints = os.path.join(SAVE_CHECKPOINTS, 'sentiment_classifier.p')
    embedding_size = 256
    units = 512
    layers = 4
    batch_size = 1
    train_path = os.path.join('vgmidi', 'labelled', 'train', 'train_dataset.csv')
    test_path = os.path.join('vgmidi', 'labelled', 'test', 'test_dataset.csv')
    layer_index = 4

    # load vocabulary
    with open(vocabulary_path) as input_file:
        vocabulary = json.load(input_file)

    # vocabulary size
    vocabulary_size = len(vocabulary)

    # rebuild generative model from checkpoint
    generative_model = build_generative_model(vocabulary_size, embedding_size, units, layers, batch_size)
    generative_model.load_weights(tf.train.latest_checkpoint(model_checkpoints))
    generative_model.build(tf.TensorShape([1, None]))

    # build datasets
    train_dataset = build_dataset(train_path, generative_model, vocabulary, layer_index)
    test_dataset = build_dataset(test_path, generative_model, vocabulary, layer_index)

    # train classifier
    sentiment_neurons, score = train_classifier_model(train_dataset, test_dataset)

    # results
    print(f'Total neurons used: {len(sentiment_neurons)}')
    print('Sentiment neurons:')
    print(sentiment_neurons)
    print(f'Model Accuracy: {score}')


if __name__ == '__main__':
    main()
