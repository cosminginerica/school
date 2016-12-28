from NeuralNet import NeuralNetwork
from NeuronLayer import NeuronLayer
import cProfile

if __name__ == "__main__":
    l1 = NeuronLayer((28, 28), True, False)
    l2 = NeuronLayer((100,))
    l3 = NeuronLayer((10,), False, True)
    network = NeuralNetwork()
    network.add_layer(l1)
    network.add_layer(l2)
    network.add_layer(l3)
    network.connect_layers()
    pr = cProfile.Profile()
    pr.enable()
    network.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    network.load_test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    network.SGD(0.1, 0.1, 30, 10)
    pr.disable()
    pr.print_stats(sort="cumtime")
