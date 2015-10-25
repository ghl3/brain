
import argparse

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


    args = parser.parse_args()

    run(args)

def run(args):

    mndata = MNIST(args.training_data)

    print "Loading the training data"
    training_data, training_labels = mndata.load_training()

    nn = Network(args.shape)

    fitted, epochs = nn.SGD(training_data, training_labels,
                            epochs=args.epochs,
                            mini_batch_size=args.mini_batch_size,
                            eta=args.eta)

    if args.testing_data:
        print "Testing data"
        test_data, test_labels = mndata.load_testing()
        evaluation = fitted.evaluate(test_data, test_labels)
        print evaluation

    if args.save:

        target_dir = mkdir_or_temp(args.save)

        fitted_path = "{}/nn.pkl".format(target_dir)

        with open(fitted_path, 'wb') as handle:
            pickle.dump(fitted, handle)

        if epochs is not None:
            for i, epoch in enumerate(epochs):
                epoch_path = '{}/nn_epoch_{}.pkl'.format(target_dir, i)
                with open(epoch_path, 'wb') as handle:
                    pickle.dump(epoch, handle)
                    print "Saved epoch {} to {}".format(i, epoch_path)


def mkdir_or_temp(dir):
    if os.path.exists(dir):
        print "Cannot save in target directory: {}.  Already exists".format(dir)
        print "Saving to dummy dir"
        handle, file_name = tempfile.mkstemp()
        return file_name

    # TODO: Race condition
    else:
        os.mkdir(dir)
        return dir


if __name__ == '__main__':
    main()
