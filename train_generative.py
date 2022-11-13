import os
import json
import numpy as np
import tensorflow as tf

import midi_encoder

SAVE_CHECKPOINTS = os.path.join('trained')

TRAIN_DATASET = os.path.join('vgmidi', 'unlabelled', 'train')
TEST_DATASET = os.path.join('vgmidi', 'unlabelled', 'test')

embedding_size = 256
units = 512
layers = 2
batch = 64
epochs = 10
sequence_len = 256
learning_rate = 0.001
dropout = 0.05


def build_vocabulary(train_vocab, test_vocab):
    # Merge train and test vocabulary
    vocab = list(train_vocab | test_vocab)
    vocab.sort()

    # Calculate vocab size
    vocab_size = len(vocab)

    # create dictionary to support indexing
    vocab_dict = {char: i for i, char in enumerate(vocab)}

    # Save vocabulary encoding as a json file for generate midi later
    with open(os.path.join(SAVE_CHECKPOINTS, "vocabulary_dict.json"), "w") as f:
        json.dump(vocab_dict, f)

    return vocab_dict, vocab_size


def build_dataset(text, vocab_dict, sequence_length, batch_size, buffer_size=10000):
    # list with indices in vocabulary
    indexed_text = np.array([vocab_dict[i] for i in text.split(' ')])
    # create tf dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(indexed_text)

    sequences = tf_dataset.batch(sequence_length, drop_remainder=True)

    dataset = sequences.map(__split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset


def build_generative_model(vocab_size, embed_size, lstm_units, lstm_layers, batch_size, drop_out=0.0):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Embedding(vocab_size,
                                  embed_size,
                                  batch_input_shape=[batch_size, None])
    )

    for layer in range(max(1, lstm_layers)):
        model.add(
            tf.keras.layers.LSTM(lstm_units,
                                 return_sequences=True,
                                 stateful=True,
                                 dropout=drop_out,
                                 recurrent_dropout=drop_out
                                 )
        )

    model.add(tf.keras.layers.Dense(vocab_size))
    return model


def generative_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train_generative_model(model, train_dataset, test_dataset, model_epochs, learn_rate):
    # add optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss=generative_loss)

    # checkpoint
    checkpoint_prefix = os.path.join(SAVE_CHECKPOINTS, "generative_checkpoint_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    return model.fit(train_dataset, epochs=model_epochs, validation_data=test_dataset, callbacks=[checkpoint_callback])


def __split_input_target(chuck):
    input_text = chuck[:-1]
    target_text = chuck[1:]
    return input_text, target_text


def main():
    train_text, train_vocab = midi_encoder.load(TRAIN_DATASET)
    test_text, test_vocab = midi_encoder.load(TEST_DATASET)

    # build vocabulary
    vocab_dict, vocab_size = build_vocabulary(train_vocab, test_vocab)

    # build dataset from encoded midis
    train_dataset = build_dataset(train_text, vocab_dict, sequence_len, batch)
    test_dataset = build_dataset(test_text, vocab_dict, sequence_len, batch)

    # build generative model
    generative_model = build_generative_model(vocab_size, embedding_size, units, layers, batch, dropout)

    # save checkpoint
    generative_model.load_weights(tf.train.latest_checkpoint(SAVE_CHECKPOINTS))

    # train model
    history = train_generative_model(generative_model, train_dataset, test_dataset, epochs, learning_rate)
    # TODO: ver se está guardado


if __name__ == "__main__":
    main()
