import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import CellCNN

MULTICELLSIZE = 1000

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = CellCNN.Dataset(MULTICELLSIZE)
        
    def test_get_multi_cell_input(self):
        ret = self.dataset.get_multi_cell_input(1000)
        self.assertEqual(ret.shape, (MULTICELLSIZE,2))


if __name__ == '__main__':
    unittest.main()    