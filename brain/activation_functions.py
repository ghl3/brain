

from functions import *


# Activation functions take the array of
# z's and convert them into an array of activations

class Sigmoid(object):

    @staticmethod
    def fn(A):
        return sigmoid(A)

    @staticmethod
    def fprime(A):
        return sigmoid_prime(A)


class SoftMax(object):

    @staticmethod
    def fn(A):

        total = 0.0
        for a in a:
            total += np.exp(a)

        return np.exp(A) / total
        #return sigmoid(x)

    @staticmethod
    def fprime(a):
        raise NotImplementedError()
        #return sigmoid_prime(a)
