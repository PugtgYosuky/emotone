import os
import json
import numpy as np
import tensorflow as tf
import midi_encoder

from train_generative import build_generative_model
from train_classifier import preprocess_sentence

GENERATED_DIR = os.path.join('generated')
SAVE_CHECKPOINTS = os.path.join('trained')
VOCABULARY_DIR = os.path.join(SAVE_CHECKPOINTS, "vocabulary_dict.json")

embedding_size = 256
units = 512
layers = 2
sequence_init = '\n'
sequence_len = 256
lstm_unit_as_encoder = -2
override_path = ''


def override_neurons(model, layer_index, override):
    h, c = model.get_layer(index=layer_index).states

    c = c.numpy()

    for neuron, value in override.items():
        c[:int(neuron)] = int(value)

    model.get_layer(index=layer_index).states = (h, tf.Variable(c))


def sample_next(predictions, k):
    # sample using a categorical distribution over the top k midi chars
    # creates the following based on the previous predictions
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < 0.5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.math.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]

    return predicted_id


def process_init_text(model, init_text, vocabulary, layer_index, override):
    model.reset_states()

    for char in init_text.split(''):
        # run a forward pass
        try:
            input_eval = tf.expand_dims([vocabulary[char]], 0)

            # override sentiment neurons
            override_neurons(model, layer_index, override)

            predictions = model(input_eval)
        except:
            if char != '':
                print('Cannot process char ', char)
    return predictions


def generate_midi(model, vocabulary, index_vocabulary, init_text="", sequence_len=256, k=3, layer_index=-2,
                  override={}):
    # add padding
    init_text = preprocess_sentence(init_text)

    # to store the results
    midi_generated = []

    # process initial text
    predictions = process_init_text(model, init_text, vocabulary, layer_index, override)

    model.reset_states()
    for i in range(sequence_len):
        # remove batch dimension
        predictions = tf.squeeze(predictions, 0).numpy()

        # predict next value
        predicted_id = sample_next(predictions, k)

        # add to the generated midi
        midi_generated.append(index_vocabulary[predicted_id])

        # override sentiment neurons
        override_neurons(model, layer_index, override)

        # run a forward pass
        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    return init_text + ' ' + ' '.join(midi_generated)


def main():
    # load vocabulary
    with open(VOCABULARY_DIR, 'r') as file:
        vocabulary = json.load(file)

    # load override dictionary
    override = {}
    try:
        with open(override_path, 'r') as over_file:
            override = json.load(over_file)
    except:
        print('Override file does not exist')

    # index vocabulary
    index_vocabulary = {index: char for char, index in vocabulary.items()}

    vocabulary_len = len(vocabulary)

    # build model from checkpoints
    model = build_generative_model(vocabulary_len, embedding_size, units, layers, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(SAVE_CHECKPOINTS))
    model.build(tf.TensorShape([1, None]))

    # generate midi as text
    midi_text = generate_midi(model, vocabulary, index_vocabulary, sequence_init, sequence_len,
                              layer_index=lstm_unit_as_encoder, override=override)

    # write midi
    midi_encoder.write(midi_text, os.path.join(GENERATED_DIR, 'generated.mid'))


if __name__ == '__main__':
    main()
