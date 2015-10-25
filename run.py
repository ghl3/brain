
import argparse

from brain import Network

from mnist import MNIST


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

    nn = Network(args.shape) #[784, 30, 10])

    fitted, iterations = nn.SGD(training_data, training_labels,
                                epochs=args.epochs,
                                mini_batch_size=args.mini_batch_size,
                                eta=args.eta)

    if args.testing_data:
        print "Testing data"
        test_data, test_labels = mndata.load_testing()
        evaluation = fitted.evaluate(test_data, test_labels)
        print evaluation

    if args.save:
        print "Too lazy to do save yet..."
        raise NotImplementedError()


if __name__ == '__main__':
    main()
