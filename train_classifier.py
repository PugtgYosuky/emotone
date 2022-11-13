import os
import json
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

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
                print(f'Cannot process char |{char}|')
    h, c = model.get_layer(index=layer_index).states

    c = tf.squeeze(c, 0)
    # h = tf.squeeze(h,0)

    return tf.math.tanh(c).numpy()


def build_dataset(midis_path, data_path,  generative_model, vocabulary, layer_index):
    x = []
    y = []

    data = pd.read_csv(data_path)
    for row_index in range(data.shape[0]):
        label = data.loc[row_index]['emotion']
        filename = data.loc[row_index]['midi']
        music_name = filename.split('.')[0]
        phrase_path = os.path.join(midis_path, filename)
        encoded_path = os.path.join(midis_path, music_name + '.npt')

        # load midi as text
        if os.path.isfile(encoded_path):
            encoding = np.load(encoded_path)
        else:
            if not os.path.exists(phrase_path):
                continue

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
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if neurons_not_zero == 1:
        neuron_ixs = np.array([np.argmax(weight_penalties)])
    elif neurons_not_zero >= np.log(len(weight_penalties)):
        neuron_ixs = np.argsort(weight_penalties)[-neurons_not_zero:][::-1]
    else:
        neuron_ixs = np.argpartition(weight_penalties, -neurons_not_zero)[-neurons_not_zero:]
        neuron_ixs = (neuron_ixs[np.argsort(weight_penalties[neuron_ixs])])[::-1]

    return neuron_ixs


def train_classifier_model(train_dataset, test_dataset, reg_strength=2 ** np.arange(-8, 1).astype(float), seed=42,
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
    best_reg_strength = reg_strength[np.argmax(scores)]
    best_reg_strength = reg_strength[np.argmax(scores)]
    final_classifier = LogisticRegression(C=best_reg_strength, penalty=penalty, random_state=seed + len(reg_strength),
                                          solver='liblinear')
    final_classifier.fit(train_x, train_y)
    score = final_classifier.score(test_x, test_y) * 100

    # save model
    with open(os.path.join(SAVE_CHECKPOINTS, 'sentiment_classifier.p'), 'wb') as file:
        pickle.dump(final_classifier, file)

    # get activated neurons

    sentiment_neurons = get_activated_neurons(final_classifier)

    # plot plots
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
    midis_path = os.path.join('vgmidi', 'labelled', 'midi')
    data_path = os.path.join('vgmidi', 'labelled', 'dataset', 'sentiment_labelled.csv')
    test_percentage = 0.2
    layer_index = 4

    # load vocabulary
    with open(vocabulary_path) as input_file:
        vocabulary = json.load(input_file)

    # vocabulary size
    vocabulary_size = len(vocabulary)

    start = time.time()
    # rebuild generative model from checkpoint
    generative_model = build_generative_model(vocabulary_size, embedding_size, units, layers, batch_size)
    generative_model.load_weights(tf.train.latest_checkpoint(SAVE_CHECKPOINTS))
    generative_model.build(tf.TensorShape([1, None]))

    # build datasets
    x, y = build_dataset(midis_path, data_path, generative_model, vocabulary, layer_index)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage)

    train_dataset = (x_train, y_train)
    test_dataset = (x_test, y_test)

    # train classifier
    sentiment_neurons, score = train_classifier_model(train_dataset, test_dataset)
    end = time.time()
    # plots
    print(f'Total neurons used: {len(sentiment_neurons)}')
    print('Sentiment neurons:')
    print(sentiment_neurons)
    print(f'Model Accuracy: {score}')
    print('Total time:', (end - start)/60, 'minutes')


if __name__ == '__main__':
    main()
