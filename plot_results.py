import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

PLOTS_DIRECTORY = os.path.join('results')


def plot_logistc_and_save(xs, ys, neuron):
    sentiment_unit = xs[:, neuron]

    plt.figure()
    plt.hist(sentiment_unit[ys == 1], bins=50, alpha=0.5, label='Joy phrases')
    plt.hist(sentiment_unit[ys == 2], bins=50, alpha=0.5, label='Sad phrases')
    plt.hist(sentiment_unit[ys == 3], bins=50, alpha=0.5, label='Relaxed phrases')
    plt.hist(sentiment_unit[ys == 4], bins=50, alpha=0.5, label='Distress phrases')
    plt.ylabel('Number of phrases')
    plt.xlabel('Value of the sentiment neuron')
    plt.legend()
    plt.title(f'Distribution of the logistic values - neuron {neuron}')
    plt.savefig(os.path.join(PLOTS_DIRECTORY, f'neuron_{neuron}.png'))
    plt.clf()


def plot_logistcs(xs, ys, top_neurons):
    for neuron in top_neurons:
        plot_logistc_and_save(xs, ys, neuron)


def plot_weight_contributions(coef):
    plt.title('Valued of resulting L! penalized weights')
    plt.tick_params(axis='both', which='major')

    # normalize weights contribution
    norm = np.linalg.norm(coef)
    coef = coef/norm

    plt.plot(range(len(coef[0])), coef.T)
    plt.xlabel('Neuron index')
    plt.ylabel('Neuron weight')
    plt.savefig(os.path.join(PLOTS_DIRECTORY, 'weight_contributions.png'))
    plt.clf()
