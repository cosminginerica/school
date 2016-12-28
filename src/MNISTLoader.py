import struct

DEBUG = True

class MNISTLoader:
    def __init__(self, data_path, labels_path):
        """
        Read MNIST data: this dataset comes in two files: one with the handwritten
        digits and another with the labels for each digit (0...9)
        :param data_path: path to file with the binary encoded digits
        :param labels_path: path to the file with the binary encoded labels
        """
        self._images = []
        self._labels = []
        self._data_path = data_path
        self._labels_path = labels_path
        self._rows = 0
        self._cols = 0
        self._num_items = 0

    def normalize(self):
        for i in range(0, len(self._images)):
            self._images[i] = [float(j) / 255 for j in self._images[i]]


    def get_data(self):
        self.normalize()
        return list(zip(self._images, self._labels))

    def read_data(self):
        with open(self._data_path, "rb") as data:
            bytes_read = data.read()

        magic_number, self._num_items, self._rows, self._cols = \
            struct.unpack(">iiii", bytes_read[:16])

        # check if we read right
        if magic_number != 2051:
            print("The data file {0} was not read right.".format(self._data_path))
            return False
        step = self._cols * self._rows

        if DEBUG is True:
            num_items = 5000
        else:
            num_items = self._num_items
        for i in range(0, num_items):  # skip first 16 bytes (header)
            self._images.append(struct.unpack("B" * step,
                                              bytes_read[16 + i * step: 16 + (i + 1) * step]))

        return True

    def read_labels(self):

        with open(self._labels_path, "rb") as data:
            bytes_read = data.read()

        magic_number, num_items = struct.unpack(">ii", bytes_read[:8])

        if DEBUG is True:
            num_i = 5000
        else:
            num_i = num_items

        # check if we read right
        if magic_number != 2049 or num_items != self._num_items:
            print("The label file {0} was not read right.".format(self._labels_path))
            return False

        self._labels.extend(struct.unpack("B" * num_i, bytes_read[8 : 8 + num_i]))
        return True
