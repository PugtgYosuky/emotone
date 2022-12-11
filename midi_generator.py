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
override_path = os.path.join(SAVE_CHECKPOINTS, 'neurons_sad.json')

embedding_size = 256
units = 4096
layers = 1
sequence_init = '\n'
sequence_len = 256
lstm_unit_as_encoder = -3


def override_neurons(model, layer_idx, override):
    h_state, c_state = model.get_layer(index=layer_idx).states

    c_state = c_state.numpy()
    for neuron, value in override.items():
        c_state[:,int(neuron)] = int(value)

    model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))


def sample_next(predictions, k):
    # Sample using a categorical distribution over the top k midi chars
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < .5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.math.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]

    return predicted_id


def process_init_text(model, init_text, char2idx, layer_idx, override):
    model.reset_states()

    for c in init_text.split(" "):
        # Run a forward pass
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)

            # override sentiment neurons
            override_neurons(model, layer_idx, override)

            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", c)

    return predictions


def generate_midi(model, char2idx, idx2char, init_text="\n ", seq_len=256, k=3, layer_idx=-3, override={}):
    # Add front and end pad to the initial text
    init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    predictions = process_init_text(model, init_text, char2idx, layer_idx, override)

    # Here batch size == 1
    model.reset_states()
    for i in range(seq_len):
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0).numpy()

        # Sample using a categorical distribution over the top k midi chars
        predicted_id = sample_next(predictions, k)

         # Append it to generated midi
        midi_generated.append(idx2char[predicted_id])

        # override sentiment neurons
        override_neurons(model, layer_idx, override)

        #Run a new forward pass
        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    return init_text + " " + " ".join(midi_generated)


def main():
    # load vocabulary
    with open(VOCABULARY_DIR, 'r') as file:
        vocabulary = json.load(file)

    # load override dictionary
    override = {}
    try:
        with open(override_path, 'r') as over_file:
            override = json.load(over_file)
    except FileNotFoundError:
        print('Override file does not exist')

    # index vocabulary
    index_vocabulary = {index: char for char, index in vocabulary.items()}

    vocabulary_len = len(vocabulary)
    print('Vocabulary len:', vocabulary_len)

    # build model from checkpoints
    print(1)
    model = build_generative_model(vocabulary_len, embedding_size, units, layers, batch_size=1)
    print(2)
    model.load_weights(tf.train.latest_checkpoint(SAVE_CHECKPOINTS))
    print(3)
    model.build(tf.TensorShape([1, None]))
    print(4)

    # generate midi as text
    midi_text = generate_midi(model, vocabulary, index_vocabulary, sequence_init, sequence_len,
                              layer_index=lstm_unit_as_encoder, override=override)

    # write midi
    midi_encoder.write(midi_text, os.path.join(GENERATED_DIR, 'sad.mid'))
    print("generated midi")


if __name__ == '__main__':
    main()
