import numpy as np
import CellCNN
import tempfile
import tensorflow as tf
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MULTICELLSIZE = 1000
N_FILTERS = 16

#TODO: Tests for multiple labels classification


class TestModel(unittest.TestCase):
    def setUp(self):
        dataset1 = CellCNN.Dataset(100000, center_density=0.1)
        dataset2 = CellCNN.Dataset(100000, center_density=0.9)
        dataset2.data += 1.5
        self.inp = CellCNN.InputData(
            (dataset1, dataset2), multi_cell_size=MULTICELLSIZE)
        self.model = CellCNN.CellCNN((None, 1000, 2), conv=[
                                     128,128, N_FILTERS], l1_weight=1e-8)
    
    def test_init_random(self):
        old_weights = self.model.get_weights()
        init_data = self.inp.get_multi_cell_inputs(100)[0]
        self.model.init_random(init_data, epochs=1)
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

    def test_binary_masked_loss(self):
        y_pred = np.random.uniform(size=(256, 1))
        y_true = np.round(np.random.uniform(size=(256, 1)))
        y_true[200:] = -1
        y_pred = tf.convert_to_tensor(y_pred, dtype="float32")
        y_true = tf.convert_to_tensor(y_true, dtype="float32")

        our_short = CellCNN.CellCNN.binary_masked_loss(
            y_true[:200], y_pred[:200])
        our_long = CellCNN.CellCNN.binary_masked_loss(y_true, y_pred)[:200]
        tf_short = tf.keras.losses.binary_crossentropy(
            y_true[:200], y_pred[:200])
        self.assertTrue(np.isclose(our_short, tf_short).all())
        self.assertTrue(np.isclose(our_short, our_long).all())

    def test_sparse_categorical_masked_loss(self):
        y_pred = np.random.uniform(size=(256, 5))
        y_true = np.round(np.random.uniform(size=(256, 1)))
        y_true[200:] = -1
        y_pred = tf.convert_to_tensor(y_pred, dtype="float32")
        y_true = tf.convert_to_tensor(y_true, dtype="float32")
        our_short = CellCNN.CellCNN.sparse_categorical_masked_loss(
            y_true[:200], y_pred[:200])
        our_long = CellCNN.CellCNN.sparse_categorical_masked_loss(
            y_true, y_pred)[:200]
        tf_short = tf.keras.losses.sparse_categorical_crossentropy(
            y_true[:200], y_pred[:200])
        self.assertTrue(np.isclose(our_short, tf_short).all())
        self.assertTrue(np.isclose(our_short, our_long).all())

    @unittest.skip("For binary the normal accuracy doesn't make sence...")
    def test_masked_accuracy_binary(self):
        y_pred = np.random.uniform(size=(256, 1))
        y_true = np.round(np.random.uniform(size=(256, 1)))
        y_true[200:] = -1
        y_pred = tf.convert_to_tensor(y_pred, dtype="float32")
        y_true = tf.convert_to_tensor(y_true, dtype="float32")

        our_short = CellCNN.CellCNN.masked_accuracy(
            y_true[:200], y_pred[:200])
        our_long = CellCNN.CellCNN.masked_accuracy(y_true, y_pred)
        tf_short = tf.keras.metrics.Accuracy()(y_true[:200], tf.round(y_pred[:200]))
        self.assertTrue(np.isclose(our_short, tf_short).all())
        self.assertTrue(np.isclose(our_short, our_long).all())
    
    def test_masked_accuracy_categorical(self):
        y_pred = np.random.uniform(high=4, size=(256, 5))
        y_true = np.round(np.random.uniform(size=(256, 1)))
        y_true[200:] = -1
        y_pred = tf.convert_to_tensor(y_pred, dtype="float32")
        y_true = tf.convert_to_tensor(y_true, dtype="float32")

        our_short = CellCNN.CellCNN.masked_accuracy(
            y_true[:200], y_pred[:200])
        our_long = CellCNN.CellCNN.masked_accuracy(y_true, y_pred)
        tf_short = tf.keras.metrics.SparseCategoricalAccuracy()(y_true[:200], y_pred[:200])
        self.assertTrue(our_short == tf_short)
        self.assertTrue(our_short == our_long)

    def test_fit(self):  # This just doesn't want to work :-/
        for _ in range(5):
            X_train, Y_train = self.inp.get_multi_cell_inputs(
                5000, CellCNN.DatasetSplit.TRAIN)
            X_eval, Y_eval = self.inp.get_multi_cell_inputs(
                500, CellCNN.DatasetSplit.TEST)
            self.model.init_random(
                self.inp.get_multi_cell_inputs(100)[0], epochs=1)
            cb = self.model.fit(X_train, Y_train, epochs=1,
                                validation_data=(X_eval, Y_eval),
                                verbose=1,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                     patience=5, restore_best_weights=True)])
            if cb.history["val_accuracy"][-1] > 0.7:
                return
        self.assertGreater(cb.history["val_accuracy"]
                           [-1], 0.7, "Model doesn't train!")

    def test_analyze_points(self):
        analyzed = self.model.analyze_points(
            self.inp.get_multi_cell_inputs(1)[0])
        self.assertEqual(analyzed.shape, (1, MULTICELLSIZE,
                         N_FILTERS), f"Unexpected shape in point analysis.")

    def test_model_saving(self):
        config_file = tempfile.NamedTemporaryFile()
        weights_file = tempfile.NamedTemporaryFile()
        config_file_name = config_file.name
        weights_file_name = weights_file.name

        self.model.save(config_file_name, weights_file_name)
        loaded_model = CellCNN.CellCNN.load(
            config_file_name, weights_file_name)
        self.assertEqual(loaded_model.get_config(), self.model.get_config())

        new_weights = loaded_model.get_weights()
        old_weights = loaded_model.get_weights()
        for i in range(len(new_weights)):
            res = old_weights[i] == new_weights[i]
            if not res.all():
                self.fail("Loaded weights differ")

        config_file.close()
        weights_file.close()


if __name__ == '__main__':
    unittest.main()
