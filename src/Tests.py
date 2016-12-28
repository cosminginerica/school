try:
    import numpy as np
except ImportError:
    print("Numpy not found tests failed")
from MNISTLoader import MNISTLoader
from NeuronLayer import NeuronLayer

def TestMNIST(train_data):
    if train_data.read_data() is False:
        return False
    if train_data.read_labels() is False:
        return False

    data = train_data.get_data()
    if len(data) != 60000:
        return False
    return True


def TestNeuronLayer(shape1, shape2):
    neuron_layer = NeuronLayer(shape1, True, False)
    if neuron_layer.get_shape() != shape1 or \
        neuron_layer.get_activations().shape != shape1 or \
        neuron_layer.get_biases().shape != shape1:
        return False

    neuron_layer2 = NeuronLayer(shape2, False, True)

    neuron_layer.set_output(neuron_layer2)
    neuron_layer2.set_input(neuron_layer)


    if neuron_layer2.get_weights().size != neuron_layer.get_num_neurons() * neuron_layer2.get_num_neurons():
        return False

    return True
