import numpy as np
import Helpers


class NeuronLayer:
    def __init__(self, shape, is_input=False, is_output=False,
                 input_layer=None, output_layer=None):
        self._is_input = is_input
        self._is_output = is_output
        self._shape = shape
        self._input_layer = input_layer
        self._output_layer = output_layer

        # init biases with random uniform distribution centered on 0
        self._biases = np.random.uniform(-0.5, 0.5, shape)
        self._num_neurons = self._biases.size

        self._nabla_b = np.zeros(self._shape)
        self._activations = np.zeros(shape)
        self._zs = np.zeros(self._shape)
        self._deltas = np.zeros(self._shape)
        self._weights = None

    def _init_weights(self):
        if self._input_layer is not None:
            # init weights with random uniform distribution centered on 0
            self._weights = np.random.uniform(-0.5, 0.5, (self._num_neurons,
                                               self._input_layer.get_num_neurons()))
            self._nabla_w = np.zeros(self._weights.shape)
    def get_activations(self):
        return self._activations


    def get_biases(self):
        return self._biases

    def set_activations(self, activations):
        self._activations = activations

    def get_shape(self):
        return self._shape

    def get_num_neurons(self):
        return self._num_neurons

    def get_weights(self):
        return self._weights

    def get_deltas(self):
        return self._deltas

    def feedforward(self):
        """
        Calculate the output of the neuron layer, based on the output of the
        previous layer
        :return: nothing
        """
        if self._is_input is False:
            prev_activations = np.reshape(self._input_layer.get_activations(),
                                          self._input_layer.get_num_neurons())
            biases = np.reshape(self._biases, self._num_neurons)
            self._zs = np.dot(self._weights, prev_activations) + biases
            self._activations = Helpers.sigmoid_vec(self._zs)

    def backpropagate(self, label):
        """
        Back propagate in order to determine the errors in the neuron
        layer (a.k.a deltas) based on the errors in the next layer
        or if the layer is the last one, calculate deltas based on the
        desired output and the actual output
        :param label: the label of the current data being analyzed
        e.g. if the current inputs of the neural net represent the digit 9,
        the label will be '9'
        :return: nothing
        """
        if self._is_output is False:
            next_weights = np.transpose(self._output_layer.get_weights())
            next_deltas = np.reshape(self._output_layer.get_deltas(),
                                     self._output_layer.get_num_neurons())
            self._deltas = np.multiply(np.dot(next_weights, next_deltas),
                                       np.reshape(Helpers.sigmoid_prime_vec(self._zs), np.dot(next_weights, next_deltas).shape))
        else:
            self._deltas = (self._activations - Helpers.desired_output(label, self._activations.size)) \
                           * Helpers.sigmoid_prime_vec(self._zs)

        if self._is_input is False:
            # update nabla_b
            self._nabla_b += self._deltas

            # update nabla_w
            self._nabla_w += np.dot(np.atleast_2d(self._deltas).T,
                                    np.atleast_2d(self._input_layer.get_activations()))

    def reset_nabla_b(self):
        self._nabla_b = np.zeros(self._shape)

    def reset_nabla_w(self):
        if self._is_input is False:
            self._nabla_w = np.zeros(self._weights.shape)

    def update_weights(self, eta, lambda_, num_samples, mini_batch_size):
        if self._is_input is False:
            self._weights = (1 - eta * lambda_ / num_samples) * self._weights - \
                            (eta / mini_batch_size) * self._nabla_w

    def update_bias(self, eta, mini_batch_size):
        self._biases -= (eta / mini_batch_size) * self._nabla_b

    def set_input(self, input_layer):
        self._input_layer = input_layer
        self._init_weights()

    def set_output(self, output_layer):
        self._output_layer = output_layer
