import random
from MNISTLoader import MNISTLoader
import time

class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._training_data = None
        self._test_data = None

    def load_data(self, data_path, label_path):
        print("Load train data")
        start = time.time()
        mnist_data = MNISTLoader(data_path, label_path)
        mnist_data.read_data()
        mnist_data.read_labels()
        self._training_data = mnist_data.get_data()
        end = time.time()
        print("Duration: {0:.2f} s".format(end - start))

    def load_test_data(self, data_path, label_path):
        print("Load test data")
        start = time.time()
        mnist_test = MNISTLoader(data_path, label_path)
        mnist_test.read_data()
        mnist_test.read_labels()
        self._test_data = mnist_test.get_data()
        end = time.time()
        print("Duration: {0:.2f} s".format(end - start))

    def connect_layers(self):
        """
        Connect al the neuron layers: call each layers' set_intput and set_output
        functions (implicitly init_weights)
        :return:
        """
        for i in range(0, len(self._layers)):
            if i < len(self._layers) - 1:
                self._layers[i].set_output(self._layers[i + 1])
            if i > 0:
                self._layers[i].set_input(self._layers[i - 1])

    def add_layer(self, layer):
        self._layers.append(layer)

    def SGD(self, eta, lambda_, epochs, mini_batch_size):
        """
        Stochastic Gradient Descend - used to minimize the cost function
        :param eta: learning rate
        :param lambda_: regularization
        :param epochs: number of epochs with which to train
        :param mini_batch_size: number of elements to apply SGD to at each passthrough0
        :return: nothing
        """
        for i in range(0, epochs):
            random.shuffle(self._training_data)
            start = time.time()
            for j in range(0, len(self._training_data) // mini_batch_size):
                self.update_mini_batch(self._training_data[j * mini_batch_size : (j + 1) * mini_batch_size],
                                       eta,
                                       lambda_,
                                       mini_batch_size)
            end = time.time()
            if self._test_data is None:
                print("Epoch {0} finished.".format(i))
            else:
                print("Epoch {0} : {1} \ {2} \ {3:.02f} s".\
                      format(i, self.accuracy(self._test_data),
                             len(self._test_data), end - start))



    def update_mini_batch(self, batch, eta, lambda_, mini_batch_size):
        for i in range(0, len(self._layers)):
            self._layers[i].reset_nabla_b()
            self._layers[i].reset_nabla_w()
        for i in range(0, len(batch)):
            self._feedforward(batch[i][0])
            self._backpropagate(batch[i][1])
            self._update_parameters(eta, lambda_, mini_batch_size)

    def accuracy(self, source):
        acc = 0
        for test_data in source:
            self._feedforward(test_data[0])
            if self._get_result() == test_data[1]:
                acc += 1
        return acc

    def _get_result(self):
        activations_output = self._layers[-1].get_activations().reshape(self._layers[-1].get_activations().size)
        max = 0
        max_pos = 0
        for i in range(0, len(activations_output)):
            if activations_output[i] > max:
                max = activations_output[i]
                max_pos = i
        return max_pos

    def _feedforward(self, network_input):
        self._layers[0].set_activations(network_input)
        for i in range(0, len(self._layers)):
            self._layers[i].feedforward()

    def _backpropagate(self, label):
        """
        Back propagate through all neuron layers, in order to compute the gradient of
        the cost functions w.r.t. the weights and biases
        :param label: the DESIRED output label of the network
        :return: nothing
        """
        for layer in self._layers[::-1]:
            layer.backpropagate(label)

    def _update_parameters(self, eta, lambda_, mini_batch_size):
        """
        Update the neural net parameters: weights and biases after the
        feed forward pass and the back propagation pass
        :param eta: learning rate
        :param lambda_: regularization
        :param mini_batch_size: size of batch on which SGD is applied
        :return: nothing
        """
        for i in range(0, len(self._layers)):
            self._layers[i].update_weights(eta,
                                           lambda_,
                                           len(self._training_data),
                                           mini_batch_size)
            self._layers[i].update_bias(eta, mini_batch_size)

