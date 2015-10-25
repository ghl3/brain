from __future__ import division

import numpy as np
import random


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """
        Return the output of the network if "a" is input.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, targets,
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

        network = self

        # We store the history of our neural networks
        networks = [network] if save_history else None

        assert(len(training_data) == len(targets))

        # TODO: Avoid unnecessary copy
        data_and_targets = zip_data_and_labels(training_data, targets)

        n = len(data_and_targets)

        print "Num training: {} Training Shape: {} Label Shape: {}".format(n, data_and_targets[0][0].shape, data_and_targets[0][1].shape)

        for j in xrange(epochs):
            # TODO: Prevent in-place shuffling
            random.shuffle(data_and_targets)
            mini_batches = [
                data_and_targets[k:k+mini_batch_size]
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

        network = Network(self.sizes)
        network.weights = weights
        network.biases = biases

        return network


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
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])

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


    def evaluate(self, test_data, test_labels, evaluator=np.argmax):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        assert(len(test_data) == len(test_labels))

        test_results = [(evaluator(self.feedforward(_data_form(x))), y)
                        for (x, y) in zip(test_data, test_labels)]

        num = len(test_data)
        num_correct = sum(int(x == y) for (x, y) in test_results)
        num_incorrect = num - num_correct
        accuracy = num_correct / num

        return {'num_testing': len(test_data),
                'num_correct': num_correct,
                'num_incorrect': num_incorrect,
                'accuracy': accuracy}

    def cost_derivative(self, output_activations, y):
         """Return the vector of partial derivatives \partial C_x /
         \partial a for the output activations."""
         return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def zip_data_and_labels(data, labels):
    """
    Convert a list of data and a list of labels
    into the form used internally by this structure.
    Currently, creates a deep copy.
    """
    assert(len(data) == len(labels))

    # TODO: Avoid unnecessary copy
    data_and_labels = [(_data_form(d), _numeric_target_to_vec(t, n=10))
                       for d, t in zip(data, labels)]

    return data_and_labels


def _numeric_target_to_vec(target, n):
    ret = np.zeros((n,1))
    ret[target] = 1
    return ret

def _data_form(input_list):
    x = np.array(input_list)
    return np.reshape(x, (len(input_list), 1))


