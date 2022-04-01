import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
import numpy as np
from ivis import Ivis
import os
import matplotlib.pyplot as plt
class MNISTGenerator(object):
    """
    Class handling data from MNIST (original / splits for ivis / dimensionaly reduced)
    """
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
             TRAIN_NEGATIVE_PATH, TEST_POSITIVE_PATH, TEST_NEGATIVE_PATH] #TODO: This is ugly :-(
    _2D_DATA_PATH = TEMP_FOLDER + "/2D_data.npy"
    _2D_LABELS_PATH = TEMP_FOLDER + "/2D_labels.npy"
    _2D_IMAGE_PATH = TEMP_FOLDER + "/2D_data.png" 
    _2D_SCATTER_PATH = TEMP_FOLDER + "/2D_scatter.png" 
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
            if self.verbose: print("Data is cached, loading data from HD.")
            self._load_data()
        else:
            if (self.verbose): print("Generating data...")
            self._generate_data()
            if (save_to_cache):
                if self.verbose: print("Saving data...")
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
        TODO: This is 1.3 GB of data. Maybe saving the original and info for recreating would be better.
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
        """
        Return negative and positive ivis splits for given data (x) and labels (y)
        """
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
        """
        Check if cache exists on disk.
        """
        for path in MNISTGenerator.PATHS:
            if os.path.exists(path):
                pass
            else:
                print("Cache wasn't found. Downloading new data.")
                return False
        return True
    def train_model(self,
                batch_size=256, 
                epochs=10,
                validation_split=0.2,
                shuffle=True,
                build_encoder=True,
                build_recoder=True):
        self.model.fit([self.x_train, self.train_positive, self.train_negative],
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                shuffle=shuffle)
        if build_encoder: self.build_encoder()
        if build_recoder: self.build_recoder()
    def build_encoder(self):
        self.encoder = self.model.get_encoder()
    def build_recoder(self):
        self.recoder = self.model.get_recoder()
    def get_2D_mnist(self,
                     load=True,
                     save=True):
        """
        Returns: (Data, labels)
        """
        if (load and os.path.exists(MNISTGenerator._2D_DATA_PATH)
                and os.path.exists(MNISTGenerator._2D_LABELS_PATH)):
            return np.load(MNISTGenerator._2D_DATA_PATH), np.load(MNISTGenerator._2D_LABELS_PATH)  
        self.x = self.encoder.predict(self.x_train)
        self.y = self.y_train
        if save:
            np.save(MNISTGenerator._2D_DATA_PATH, self.x)
            np.save(MNISTGenerator._2D_LABELS_PATH, self.y)
        return self.x,self.y
    def scatter_2D_mnist(self,
                        show=True,
                        save=True):
        if hasattr(self, "x") and hasattr(self, "y"):
            pass
        else:
            self.get_2D_mnist(False, False)
        plt.scatter(self.x[:,0], self.x[:,1], c=self.y)
        if save: plt.savefig(MNISTGenerator._2D_SCATTER_PATH)
        if show: plt.show()        
    @staticmethod
    def load_2D_mnist(load_from_hd=True, generate_on_fail=True):
        """
        Load pregenerated dimensionally reduced mnist data, or:
            if (generate_on_fail): generate them.
        Returns: (Data, labels)
        """
        if os.path.exists(MNISTGenerator._2D_DATA_PATH) and os.path.exists(MNISTGenerator._2D_LABELS_PATH) and load_from_hd:
            x = np.load(MNISTGenerator._2D_DATA_PATH)
            y = np.load(MNISTGenerator._2D_LABELS_PATH)
            return x,y
        if generate_on_fail:
            print("Generating new data...")
            mg = MNISTGenerator()
            mg.train_model()
            mg.scatter_2D_mnist()
            mg.show_encoded_plane(show=True, save_to_file=True)
            return mg.get_2D_mnist()
        raise ValueError("Mnist data has not been generated nor loaded!")


    def load_weights(path):
        raise NotImplementedError("TODO...")
    def show_encoded_plane(self,
                        columns=15*2+1,
                        rows=20*2+1,
                        show=True,
                        save_to_file=True):
        """
        BUG: If show==False plot is left hanging and then overwrites next plot.
        Show/Save 2D plane of MNIST numbers as encoded and decoded by the IVIS model.

        Arguments
        ---------
        columns : int
            Number of columns around 0 (middle column is at x=0)
            (columns-1)//2 is the right most coordinate
            should be odd
        rows : int
            Number of rows around 0 (middle row is at y=0)
            (columns-1)//2 is the right most coordinate
            should be odd
        show : bool
            Should the graph be shown to screen?
        save_to_file : bool
            Should the graph be saved to file?
        """
        if (self.verbose):
            print ("Generating plane of numbers")
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
            fig = plt.figure(figsize=(15.,15.))
            for i in range(1, columns*rows + 1):
                img = data[i-1]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img, cmap=plt.cm.binary)
                plt.axis("off")
            if save_to_file: plt.savefig(MNISTGenerator._2D_IMAGE_PATH)
            if show: plt.show()
        generate_graph(graph, self.recoder)
        show_graph(graph)        


if __name__ == "__main__":
    print(MNISTGenerator.load_2D_mnist())