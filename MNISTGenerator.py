import os

from tensorflow.python.ops.ragged.segment_id_ops import row_splits_to_segment_ids
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
from ivis import Ivis
import os
import matplotlib.pyplot as plt
class MNISTGenerator(object):
    TEMP_FOLDER = "temp"
    X_TRAIN_PATH = TEMP_FOLDER + "/x_train.npy"
    X_TEST_PATH = TEMP_FOLDER + "/x_test.npy"
    Y_TRAIN_PATH = TEMP_FOLDER + "/y_train.npy"
    Y_TEST_PATH = TEMP_FOLDER + "/y_test.npy"
    TRAIN_POSITIVE_PATH = TEMP_FOLDER + "/train_positive.npy"
    TRAIN_NEGATIVE_PATH = TEMP_FOLDER + "/train_negative.npy"
    TEST_POSITIVE_PATH = TEMP_FOLDER + "/test_positive.npy"
    TEST_NEGATIVE_PATH = TEMP_FOLDER + "/test_negative.npy"
    PATHS = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, TRAIN_POSITIVE_PATH,
             TRAIN_NEGATIVE_PATH, TEST_POSITIVE_PATH, TEST_NEGATIVE_PATH] #This is ugly :-(
    def __init__(self,
                 save_to_cache=True,
                 load_from_cache=True,
                 verbose = True):
        self.verbose = verbose
        self._build_data(load_from_cache, save_to_cache)
        self.model = Ivis()
    def _build_data(self, load_from_cache, save_to_cache):
        """Get dataset (X/y_train/test, positive/negative) and save it to this classes variables.

        Parameters
        ----------
        load_from_cache : bool
            If there are available saved data should they be loaded from cache?
        save_to_cache : bool
            If data is newly generated, should it be saved to cache?
        """
        if (load_from_cache and self._cache_exists()):
            self._load_data()
        else:
            if (self.verbose):
                self._generate_data()
            if (save_to_cache):
                self.save_data()
    def _generate_data(self):
        """Generate new data by downloading it and then splitting.
        This is pretty time consuming (10s of seconds), so it's used when there
        is nothing in cache or new data is needed.
        If data is in cache, use `self._load_data`
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        # Scale images to the [0, 1] range
        self.x_train = self.x_train.astype("float64") / 255
        self.x_test = self.x_test.astype("float64") / 255
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)
        self.train_negative, self.train_positive = MNISTGenerator.generate_positive_negative(self.x_train, self.y_train)
        self.test_negative, self.test_positive = MNISTGenerator.generate_positive_negative(self.x_test, self.y_test)
    def _load_data(self):
        """Load data from cache.
        If there is no data to be loaded, use `self._generate_data`.
        """
        self.x_train = np.load(MNISTGenerator.X_TRAIN_PATH)
        self.x_test = np.load(MNISTGenerator.X_TEST_PATH)
        self.y_train = np.load(MNISTGenerator.Y_TRAIN_PATH)
        self.y_test = np.load(MNISTGenerator.Y_TEST_PATH)
        self.train_positive = np.load(MNISTGenerator.TRAIN_POSITIVE_PATH)
        self.train_negative = np.load(MNISTGenerator.TRAIN_NEGATIVE_PATH)
        self.test_postitive = np.load(MNISTGenerator.TEST_POSITIVE_PATH)
        self.test_negative = np.load(MNISTGenerator.TEST_NEGATIVE_PATH)
    def save_data(self):
        """Save data to cache, which can be then loaded back
        by `self._load_data`.
        """
        if (not os.path.exists(MNISTGenerator.TEMP_FOLDER)): os.mkdir(MNISTGenerator.TEMP_FOLDER)
        # TODO: Create folder, if it doesnt exist.
        np.save(MNISTGenerator.X_TRAIN_PATH, self.x_train)
        np.save(MNISTGenerator.X_TEST_PATH, self.x_test)
        np.save(MNISTGenerator.Y_TRAIN_PATH, self.y_train)
        np.save(MNISTGenerator.Y_TEST_PATH, self.y_test)
        np.save(MNISTGenerator.TRAIN_POSITIVE_PATH, self.train_positive)
        np.save(MNISTGenerator.TRAIN_NEGATIVE_PATH, self.train_negative)
        np.save(MNISTGenerator.TEST_POSITIVE_PATH, self.test_positive)
        np.save(MNISTGenerator.TEST_NEGATIVE_PATH, self.test_negative)
    def delete_cache(self):
        for path in MNISTGenerator.PATHS:
            if os.path.exists(path):
                os.remove(path)
            else:
                if (self.verbose): print(f"File \"{path}\" does not exist.")
    @staticmethod
    def generate_positive_negative(x, y):
        positive = np.zeros(shape=x.shape)
        negative = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            pos=np.nonzero(y==y[i])
            pos=np.random.choice(pos[0],1)
            positive[i,:,:,:]=x[pos,:,:,:]
            pos=np.nonzero(y!=y[i])
            pos=np.random.choice(pos[0],1)
            negative[i,:,:,:]=x[pos,:,:,:]
        return negative, positive
    def _cache_exists(self):
        for path in MNISTGenerator.PATHS:
            if os.path.exists(path):
                pass
            else:
                print("Cache wasn't found. Downloading new data.")
                return False
        return True
    def train_model(self,
                batch_size=256, 
                epochs=1,
                validation_split=0.2,
                shuffle=True,
                build_encoder=True,
                build_recoder=True):
        self.model.fit([self.x_train, self.train_positive, self.train_negative],
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                shuffle=shuffle)
        self.build_encoder()
        self.build_recoder()
    def build_encoder(self):
        self.encoder = self.model.get_encoder()
    def build_recoder(self):
        self.recoder = self.model.get_recoder()
    def get_2D_mnist(self):
        return self.encoder.predict(self.x_train)
    def show_encoded_plane(self,
                        columns=15*2+1,
                        rows=20*2+1):
        column_offset = columns//2
        row_offset = rows//2
        graph = np.zeros((columns*rows, 28, 28))

        def generate_graph(target, model):
        # Fill out given target with results from model
        # TODO: Make this more efficient. Just run one predict for all values.
            for i in range(rows):
                for j in range(columns):
                    tensor = tf.constant((j-column_offset, row_offset-i), dtype="float64")
                    target[i*columns + j] = model.predict(np.array((tensor,)))

        def show_graph(data):
        # Show graph from given from
            fig = plt.figure(figsize=(15.,15.))
            for i in range(1, columns*rows + 1):
                img = data[i-1]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img, cmap=plt.cm.binary)
                plt.axis("off")
            plt.show()
        generate_graph(graph, self.recoder)
        show_graph(graph)        


mg = MNISTGenerator()
mg.train_model()
mg.show_encoded_plane()