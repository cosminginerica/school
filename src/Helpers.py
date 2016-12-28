# Miscellaneous functions
try:
    import numpy as np
except ImportError:
    print("Numpy not found please install using pip")

def sigmoid(z):
    return 1.0 / (1.0 + np.e**(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def sigmoid_vec(z):
    s_v = np.vectorize(sigmoid)
    return s_v(z)

def sigmoid_prime_vec(z):
    s_p_v = np.vectorize(sigmoid_prime)
    return s_p_v(z)


"""
Convert a label (e.g. 9) to the equivalent in 'array-format'
- e.g. a label 9 from a range of [0, 9] will translate into [0 0 0 0 0 0 0 0 0 1]
- ideally, the nineth neuron on the layer should have the activation 1 as opposed
  to the reast of the neurons, which should have the activation 0
"""


def desired_output(label, num_outputs):
    array = np.zeros(num_outputs)
    array[label] = 1.0
    return array
