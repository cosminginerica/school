import cProfile
import os

from NeuralNet import NeuralNetwork
from NeuronLayer import NeuronLayer


MAIN_MODULE_PATH = os.path.dirname(__file__)

def main():
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
    
    training_images = os.path.abspath(os.path.join(MAIN_MODULE_PATH, "..", "data", "train-images.idx3-ubyte"))
    training_labels = os.path.abspath(os.path.join(MAIN_MODULE_PATH, "..", "data", "train-labels.idx1-ubyte"))
    
    network.load_data(training_images, training_labels)
    
    test_images = os.path.join(MAIN_MODULE_PATH, "..", "data", "t10k-images.idx3-ubyte")
    test_labels = os.path.join(MAIN_MODULE_PATH, "..", "data", "t10k-labels.idx1-ubyte")
    
    network.load_test_data(test_images, test_labels)
    
    network.SGD(0.1, 0.1, 30, 10)
    
    pr.disable()
    pr.print_stats(sort="cumtime")
    
if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception, ex:
        print("Errors occured while running neuralnet ", ex)
        exit(1)
   
