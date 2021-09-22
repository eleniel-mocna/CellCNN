import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import CellCNN

MULTICELLSIZE = 1000
N_FILTERS = 16

#TODO: Tests for multiple labels classification

class TestModel(unittest.TestCase):
    def setUp(self):
        dataset1 = CellCNN.Dataset(100000, center_density=0.1)
        dataset2 = CellCNN.Dataset(100000, center_density=0.9)
        self.inp = CellCNN.InputData(
            (dataset1, dataset2), multi_cell_size=MULTICELLSIZE)
        self.model = CellCNN.CellCNN((None, 1000, 2), conv=[64, N_FILTERS], l1_weight=0)
        
    def test_init_random(self):
        old_weights = self.model.get_weights()
        init_data = self.inp.get_multi_cell_inputs(100)[0]
        self.model.init_random(init_data)
        new_weights = self.model.get_weights()
        self.assertEqual(len(old_weights), len(
            new_weights), "Weights shape mismatch!")
        res = []
        for i in range(len(old_weights)):
            arr = old_weights[i] == new_weights[i]
            res.append(arr.any())
        for value in res:
            if value == False:
                return True
        self.fail("init_random didn't change weights")

    def test_binary_accuracy(self):
        for _ in range(100):
            y_t = tf.random.uniform((1000,))
            y_p = tf.random.uniform((1000,))
            self.assertTrue(bool(tf.keras.metrics.BinaryAccuracy()(
                y_t, y_p) == CellCNN.CellCNN.binary_accuracy(y_t, y_p)))
            y1 = tf.zeros((1000,))
            y2 = y1+1
            self.assertFalse(bool(CellCNN.CellCNN.binary_accuracy(
                y1, y1) == CellCNN.CellCNN.binary_accuracy(y1, y2)))

    @unittest.skip("Test not implemented yet...")
    def test_binary_masked_loss(self):
        pass

    @unittest.skip("Test not implemented yet...")
    def test_sparse_categorical_masked_loss(self):
        pass

    @unittest.skip("Test not implemented yet...")
    def test_masked_accuracy(self):
        pass

    def test_fit(self): # This just doesn't want to work :-/
        X_train, Y_train = self.inp.get_multi_cell_inputs(
            20000, CellCNN.DatasetSplit.TRAIN)
        X_eval, Y_eval = self.inp.get_multi_cell_inputs(
            2000, CellCNN.DatasetSplit.TEST)
        self.model.init_random(self.inp.get_multi_cell_inputs(100)[0])
        cb = self.model.fit(X_train, Y_train, epochs=1,
                            validation_data=(X_eval, Y_eval), verbose=1)
        print(self.model(X_eval)[:10])
        print(Y_eval[:10])
        print(cb.history)
        self.assertGreater(cb.history["val_accuracy"][-1], 0.7, "Model doesn't train!")

    def test_analyze_points(self):
        analyzed = self.model.analyze_points(
            self.inp.get_multi_cell_inputs(1)[0])
        self.assertEqual(analyzed.shape, (1, MULTICELLSIZE,
                         N_FILTERS), f"Unexpected shape in point analysis.")
    @unittest.skip("Need to figure out temporary files")
    def test_model_saving(self):
        pass

if __name__ == '__main__':
    unittest.main()
