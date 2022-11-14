import os
import json
import argparse
import pickle
import time

import numpy as np
import tensorflow as tf

from midi_generator import generate_midi
from train_classifier import encode_sentence, get_activated_neurons
from train_generative import build_generative_model

SAVE_CHECKPOINTS = os.path.join('trained')

GEN_MIN = -1
GEN_MAX = 1


def apply_mutation(individual, mutation_prob):
    for gene in range(len(individual)):
        if np.random.random() < mutation_prob:
            individual[gene] = np.random.uniform(GEN_MIN, GEN_MAX)
    return individual


def crossover(parent_a, parent_b):
    split_index = np.random.randint(len(parent_a))
    new_a = parent_a.copy()
    new_b = parent_b.copy()
    new_a[split_index:] = parent_b[split_index:]
    new_b[split_index:] = parent_a[split_index:]
    return new_a, new_b


def reproduce(population, new_population_size, genome_size, mutation_prob):
    new_population = np.zeros((new_population_size, genome_size))

    # in each iteration we select the parents randomly and apply the crossover
    for i in range(0, new_population_size, 2):
        parent_a = np.random.randint(len(population))
        parent_b = np.random.randint(len(population))
        new_a, new_b = crossover(population[parent_a], population[parent_b])
        new_population[i] = new_a
        new_population[i + 1] = new_b

    print('Mutation', type(mutation_prob))
    print('pop', type(new_population))
    return np.apply_along_axis(apply_mutation, 1, new_population, mutation_prob)


def roulette_wheel(population, fitness):
    # normalize fitness
    norm_fitness = fitness / np.sum(fitness)

    # total fitness to achieve
    r = np.random.uniform(0, 1)

    total_fitness = 0
    for i in range(len(population)):
        total_fitness += norm_fitness[i]

        if r < total_fitness:
            return population[i]
    return population[-1]


def selection(population, fitness, coupling_size, genome_size, elite_prob):
    coupling_pool = np.zeros((coupling_size, genome_size))

    # selection of individuals
    for i in range(coupling_size):
        coupling_pool[i] = roulette_wheel(population, fitness)

    # elitism
    elite_size = int(np.ceil(elite_prob * len(population)))
    elite_indexes = np.argsort(-fitness)

    for i in range(elite_size):
        r = np.random.randint(0, coupling_size)
        coupling_pool[r] = elite_indexes[i]

    return coupling_pool


def calculate_fitness(individual, generative_model, classifier_model, vocabulary, indexes_vocabulary, layer_index,
                      sentiment, runs=30):
    encoding_size = generative_model.layers[layer_index].units
    generated_midis = np.zeros((runs, encoding_size))

    # get activated neurons
    sentiment_neurons_indexes = get_activated_neurons(classifier_model)
    print(len(individual) == len(sentiment_neurons_indexes))

    override = {}
    for i, index in enumerate(sentiment_neurons_indexes):
        override[index] = individual[i]

    # generate pieces and encode them
    for i in range(runs):
        midi_text = generate_midi(generative_model, vocabulary, indexes_vocabulary, sequence_length=64,
                                  layer_index=layer_index, override=override)
        generated_midis[i] = encode_sentence(generative_model, midi_text, vocabulary, layer_index)

    # predict the sentiment of the generated midis
    midis_sentiment = classifier_model.predict(generated_midis).clip(min=0)

    # calculate error
    # since we have 4 sentiments instead of the 2 used by Ferreira et al., the error is calculated comapring
    # the predicted sentiment with the original sentiment
    error = np.sum(sentiment == midis_sentiment) / runs
    return 1.0 - error


def evaluate(population, generative_model, classifier_model, vocabulary, indexes_vocabulary, layer_index, sentiment):
    fitness = np.zeros((len(population), 1))

    for i in range(len(population)):
        fitness[i] = calculate_fitness(population[i], generative_model, classifier_model, vocabulary,
                                       indexes_vocabulary, layer_index, sentiment)
    return fitness


def evolve(population_size, genome_size, generative_model, classifier_model, vocabulary, indexes_vocabulary,
           layer_index, sentiment, mutation_prob, elitism_prob, epochs):
    # create initial population
    population = np.random.uniform(GEN_MIN, GEN_MAX, (population_size, genome_size))

    # evaluate initial population
    fitness_population = evaluate(population, generative_model, classifier_model, vocabulary, indexes_vocabulary,
                                  layer_index, sentiment)
    print(f'--> Fitness: {fitness_population}')
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        # selection
        selected_individuals = selection(population, fitness_population, population_size, genome_size, elitism_prob)

        # reproduce with parent with crossover and mutation
        population = reproduce(selected_individuals, population_size, genome_size, mutation_prob)

        # calculate the new fitness
        fitness_population = evaluate(population, generative_model, classifier_model, vocabulary, indexes_vocabulary,
                                      layer_index, sentiment)
        print(f'--> Fitness: {fitness_population}')

    return population, fitness_population


if __name__ == '__main__':
    # arguments
    gen_model_path = os.path.join('trained')
    clf_model_path = os.path.join('trained', 'sentiment_classifier.p')
    vocabulary_path = os.path.join('trained', 'vocabulary_dict.json')
    layer_index = 1
    elitism = 0.1
    epochs = 10
    layers = 2
    units = 512
    embed = 256
    population_size = 10
    mutation = 0.2

    sentiment = 2  # ! CHANGE THE SENTIMENT

    # load vocabulary
    with open(vocabulary_path) as file:
        vocabulary = json.load(file)

    # create indexes vocabulary
    indexes_vocabulary = {index: char for char, index in vocabulary.items()}

    vocabulary_size = len(vocabulary)

    # start counting time
    start = time.time()
    # TODO: check if its equal to the used in the creation
    # load generative model
    generative_model = build_generative_model(vocabulary_size, embed, units, layers, batch_size=1)
    generative_model.load_weights(tf.train.latest_checkpoint(gen_model_path))
    generative_model.build(tf.TensorShape([1, None]))

    # load classifier model
    with open(clf_model_path, 'rb') as file:
        classifier_model = pickle.load(file)

    sentiment_neuron_indexes = get_activated_neurons(classifier_model)
    genome_size = len(sentiment_neuron_indexes)

    # evolve(population_size, genome_size, generative_model, classifier_model, vocabulary, indexes_vocabulary,
    #            layer_index, sentiment, mutation_prob, elitism_prob, epochs):
    population, fitness = evolve(population_size, genome_size, generative_model, classifier_model, vocabulary,
                                 indexes_vocabulary, layer_index, sentiment, mutation, elitism, epochs)

    # best individual
    best_index = np.argmax(fitness)
    best_individual = population[best_index]

    # use the best individual to create a dictionary
    neurons = {}
    for i, index in enumerate(sentiment_neuron_indexes):
        neurons[str(index)] = best_individual[i]
    print('Neurons: \n:', neurons)

    # save dictionary
    sentiment_class = {1: 'joy', 2: 'sad', 3: 'relaxed', 4: 'distress'}

    save_path = os.path.join('trained', f'neurons_{sentiment_class[sentiment]}.json')
    with open(save_path, 'w') as file:
        json.dump(neurons, file)

    end = time.time()
    print('Total time:', (end - start)/60, 'seconds')
