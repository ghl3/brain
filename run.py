
import argparse

import numpy as np

from brain import Network

from mnist import MNIST

import pickle

import os
import tempfile


def main():

    parser = argparse.ArgumentParser(description='Train a neural network.')

    parser.add_argument("--shape", metavar='N', type=int, required=True, nargs='+',
                        help='Shape of the neural network.  One should pass a list of integers: "--shape 30 10 1')

    parser.add_argument("--training_data", type=str, default="./data",
                        help="Location of a directory containing the training data")

    parser.add_argument("--testing_data", type=str, default="./data",
                        help="Location of a directory containing the testing")

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for the training')

    parser.add_argument('--mini_batch_size', type=int, default=10,
                        help='Number of training rows in each mini-batch')

    parser.add_argument('--eta', type=float, default=3.0,
                        help='Learning rate for the neural network')

    parser.add_argument('--save', type=str, default=None,
                        help='If present, directory where the results should be saved.  Will not overwrite existing directory.')

    parser.add_argument('--save-epochs', default=False, action='store_true',
                        help='Save a checkpoint of the model at all epochs.')

    args = parser.parse_args()

    run(args)


def run(args):

    mndata = MNIST(args.training_data)

    print "Loading the training data"
    training_data, training_labels = mndata.load_training()

    training_data = convert_training_data(training_data)
    training_labels = convert_number_labels_to_vectors(training_labels)

    nn = Network(args.shape)

    fitted, epochs = nn.SGD(training_data, training_labels,
                            epochs=args.epochs,
                            mini_batch_size=args.mini_batch_size,
                            eta=args.eta,
                            save_history=args.save_epochs)

    if args.testing_data:
        print "Testing data"
        test_data, test_labels = mndata.load_testing()
        test_data = convert_training_data(test_data)
        # For evaluation, we put the index of the label
        # with the argmax
        evaluation = fitted.evaluate(test_data, test_labels,
                                     evaluator=np.argmax)
        print evaluation

    if args.save:

        label_dir = mkdir_or_temp(args.save)

        fitted_path = "{}/nn.pkl".format(label_dir)

        with open(fitted_path, 'wb') as handle:
            pickle.dump(fitted, handle)

        if epochs is not None:
            for i, epoch in enumerate(epochs):
                epoch_path = '{}/nn_epoch_{}.pkl'.format(label_dir, i)
                with open(epoch_path, 'wb') as handle:
                    pickle.dump(epoch, handle)
                    print "Saved epoch {} to {}".format(i, epoch_path)


def mkdir_or_temp(dir):
    if os.path.exists(dir):
        print "Cannot save in label directory: {}.  Already exists".format(dir)
        print "Saving to dummy dir"
        handle, file_name = tempfile.mkstemp()
        return file_name

    # TODO: Race condition
    else:
        os.mkdir(dir)
        return dir


def _numeric_label_to_vec(label, n):
    ret = np.zeros((n,1), dtype='float32')
    ret[label] = 1.0
    return ret


def convert_training_data(training_data):
    return [_data_form(d) for d in training_data]


def convert_number_labels_to_vectors(labels):
    return [_numeric_label_to_vec(l, 10) for l in labels]


def _data_form(input_list):
    x = np.array(input_list, dtype='float32')
    x = x / 256.0
    return np.reshape(x, (len(input_list), 1))


if __name__ == '__main__':
    main()
