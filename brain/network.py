from __future__ import division

import numpy as np

from cost_functions import *
from functions import *

class Network(object):

    def __init__(self, sizes, weights=None, biases=None,
                 cost = QuadraticCost(),
                 seed=None):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost

        self._np_random = np.random.RandomState(seed)

        if biases is not None:
            self.biases = biases
        else:
            self.biases = [self._np_random.randn(y, 1)
                            for y in sizes[1:]]

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [self._np_random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """
        Return the output of the network if "a" is input.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, labels,
            epochs, mini_batch_size, eta,
            save_history=False):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.
        Returns an updated neural network (this function does NOT
        update the current neural network in place).
        Optionally, return a history of the network after every
        batch.
        """

        assert(len(training_data) == len(labels))
        assert(training_data[0].shape == (self.sizes[0], 1))
        assert(labels[0].shape == (self.sizes[-1], 1))

        data_and_labels = zip(training_data, labels)

        n = len(data_and_labels)

        network = self

        # We store the history of our neural networks
        networks = [network] if save_history else None

        print "Num training: {} Training Shape: {} Label Shape: {}".format(n, data_and_labels[0][0].shape, data_and_labels[0][1].shape)

        for j in xrange(epochs):
            shuffled_data = self.randomly_ordered(data_and_labels)
            mini_batches = [
                shuffled_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            for i, mini_batch in enumerate(mini_batches):
                network = network.update_mini_batch(mini_batch, eta)

                if i % 1000 == 0:
                    print "Mini Batch: {} in Epoch {} complete".format(i, j)

            # We save at the end of an epoch
            if save_history:
                networks.append(network)

        return network, networks


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        weights = [w-(eta/len(mini_batch))*nw
                   for w, nw in zip(self.weights, nabla_w)]

        biases = [b-(eta/len(mini_batch))*nb
                  for b, nb in zip(self.biases, nabla_b)]

        return Network(self.sizes, weights=weights, biases=biases)


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x

        # list to store all the activations, layer by layer
        activations = [x]

        # list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y) #   _function_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, data, label_indices, evaluator=np.argmax):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        assert(len(data) == len(label_indices))

        assert(len(data) == len(label_indices))
        assert(data[0].shape == (self.sizes[0], 1))
        for idx in label_indices:
            assert(idx >= 0 and idx < self.sizes[-1])

        results = [(evaluator(self.feedforward(x)), y)
                        for (x, y) in zip(data, label_indices)]

        num = len(data)
        num_correct = sum(int(x == y) for (x, y) in results)
        num_incorrect = num - num_correct
        accuracy = num_correct / num

        return {'num_testing': len(data),
                'num_correct': num_correct,
                'num_incorrect': num_incorrect,
                'accuracy': accuracy}


    def random_order(self, n):
        indices = range(n)
        self._np_random.shuffle(indices)
        return indices


    def randomly_ordered(self, items):
        return [items[i] for i in self.random_order(len(items))]


