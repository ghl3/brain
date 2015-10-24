

from brain import Network

from mnist import MNIST


def main():

    mndata = MNIST('./data')

    print "Loading the training data"
    training_data, training_labels = mndata.load_training()

    nn = Network([784, 30, 10])

    fitted = nn.SGD(training_data, training_labels, 30, 10, 3.0)


if __name__ == '__main__':
    main()
