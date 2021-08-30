import CellCNN
import tensorflow as tf
import numpy as np
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MULTICELLSIZE = 1000


class TestInputData(unittest.TestCase):
    def setUp(self):
        dataset1 = CellCNN.Dataset(10000, center_density=0.01)
        dataset2 = CellCNN.Dataset(10000, center_density=0.99)
        self.inp = CellCNN.InputData(
            (dataset1, dataset2), multi_cell_size=MULTICELLSIZE)
    
    def test_length(self):
        self.assertEqual(self.inp.length, 2)
    def test_get_multi_cell_inputs(self):
        AMOUNT=250
        x,y = self.inp.get_multi_cell_inputs(amount=AMOUNT)
        self.assertEqual(x.shape, (AMOUNT, MULTICELLSIZE, 2))
        self.assertEqual(len(y), 1)
        self.assertEqual(len(y[0]), AMOUNT)

class TestInputDataMultiLabel(unittest.TestCase):
    def setUp(self):

        dataset1 = CellCNN.Dataset(10000, center_density=0.01)
        dataset2 = CellCNN.Dataset(10000, center_density=0.99)
        labels=np.array(((0, 2, 10), (1, 0, 1)))
        self.NLABELS = 3
        self.inp = CellCNN.InputData(
            (dataset1, dataset2), multi_cell_size=MULTICELLSIZE, labels=labels)
    def test_dimensions(self):
        self.assertEqual(self.inp.length, 2)
        AMOUNT=250
        x,y = self.inp.get_multi_cell_inputs(amount=AMOUNT)
        self.assertEqual(x.shape, (AMOUNT, MULTICELLSIZE, 2))
        self.assertEqual(len(y), self.NLABELS)
        self.assertEqual(len(y[0]), AMOUNT)


if __name__ == '__main__':
    unittest.main()
