import json
from os import name
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Lambda, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine import training
from tensorflow.python.ops.math_ops import reduce_sum

#TODO: Bugfix masking.


class L1Layer(Layer):
    """Layer which passes values through and applies
    1L regularization to them.
    """

    def __init__(self, loss_weight=0.01):
        """Initialize the layer.

        Parameters
        ----------
        loss_weight : float, optional
            How strong should the regularization be, by default 0.01
        """
        super(L1Layer, self).__init__()
        self.loss_weight = loss_weight

    def call(self, inputs):
        self.add_loss(self.loss_weight * reduce_sum(inputs))
        return inputs


class CellCNN(Model):
    """Implementation of CellCNN.

    Implementation of `CellCNN network<https://www.nature.com/articles/ncomms14825>`_.
    
    With some added functionalities such as using multiple layers
    of filters, using multiple sets of labels or applying
    l1 regularization to filter outputs.
    """

    def __init__(self,
                 input_shape,
                 classes=[2, ],
                 conv=[16, ],
                 k=25,
                 lr=0.01,
                 activation="relu",
                 l1_weight=0.01,
                 dropout=0.25):
        """Initialize CellCNN model.

        Parameters
        ----------
        input_shape : tuple (None, ncell, dims)
            Description of input shape, where:
            input_shape[0]=None: for batch size,
            input_shape[1]: n of cells in a multi_cell input,
            input_shape[2]: n of dimensions for every cell.
        classes : list, optional
            Description of labels used, where:
            0: linear variable,
            2: binary classification,
            n > 2: n-nary classification.
            E.g. [2,0] expects labels to express a binary classification
            problem and a regression problem.
            By default [2, ]
        conv : list, optional
            Number of filters in convolution layers,
            e.g. [128, 128, 16] describes two layers of 
            128 nodes followed by a layer of 16 nodes.
            By default [16, ]
        k : int, optional
            Number of cells that go through pooling
            after the last filter layer, 
            by default 25
        lr : float, optional
            Learning rate, by default 0.01.
        activation : str or tf activation function, optional
            Activation function, by default "relu"
        l1_weight : float, optional
            Weight of l1 regularization applied to the last
            filter outputs, 0 for no regularization,
            by default 0.01.
        dropout : float, optional
            Strength of dropout before every filter layer,
            by default 0.25
        """
        super(CellCNN, self).__init__()
        self.my_input_shape = input_shape
        self.classes = classes
        self.conv = conv
        self.k = k
        self.lr = lr
        self.activation = activation
        self.l1_weight = l1_weight
        self.dropout = dropout
        self._build_layers()
        self._compile()
        self.build(tuple(self.my_input_shape))

    def _build_layers(self):
        """Build layers for arguments given in __init__()

        Used by __init__.
        """
        self._build_filter_layers()
        self._build_output_layers()

    def _build_filter_layers(self):
        """Build the first portion of layers, mainly responsible for filtering.

        Used by _build_layers.
        """
        self.my_layers = []
        for i in range(len(self.conv)):
            self.my_layers.append(Dropout(self.dropout))
            if self.conv[i] != 0:
                self.my_layers.append(
                    Conv1D(filters=self.conv[i],
                           kernel_size=1,
                           activation=self.activation,
                           )
                )

        self.my_layers.append(L1Layer(self.l1_weight))
        self.my_layers.append(Lambda(self._select_top,
                                     output_shape=(1,),
                                     name="pooling"
                                     )
                              )

    def _build_output_layers(self):
        """Build the second portion of layers, mainly responsible for generating output

        Used by _build_layers.
        
        Raises
        ------
        ValueError
            Not supported label description given.
        """
        self.output_layers = []
        self.loss_functions = []

        k = 1
        for i in self.classes:
            layer_name = "output_" + str(k)
            if i == 0:
                self.output_layers.append(Dense(1, name=layer_name))
                self.loss_functions.append(CellCNN.mse_masked_loss)
                # self.loss_functions[layer_name] = "mse"
            elif i == 2:
                self.output_layers.append(
                    Dense(1, activation="sigmoid", name=layer_name))
                self.loss_functions.append(CellCNN.binary_masked_loss)
                #self.loss_functions.append(
                #    tf.keras.losses.BinaryCrossentropy())
            elif i > 2:
                self.output_layers.append(
                    Dense(i, activation="softmax", name=layer_name))
                #self.loss_functions.append(
                #    tf.keras.losses.SparseCategoricalCrossentropy())
                self.loss_functions.append(
                    CellCNN.sparse_categorical_masked_loss)
            else:
                raise ValueError("Invalid output layer specification given!")
            k += 1

    def _compile(self):
        """Compile the model.

        Used by __init__.
        """
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=self.loss_functions,
            metrics=[CellCNN.masked_accuracy,
                     'accuracy',
                     CellCNN.binary_accuracy
                     ]
        )

    def init_random(self,
                    data,
                    labels=None,
                    n_classes=10,
                    epochs=10,
                    batch_size=256):  # THIS CAN'T WORK!!! BUG
        init_model = InitCellCNN.load_from_dict(self.get_config(), n_classes)
        if labels is None:
            labels = np.random.randint(
                low=n_classes, size=data.shape[0], dtype='l')  # Does this work?

        init_model.fit(data, labels, batch_size=batch_size, epochs=epochs)
        for i in range(len(init_model.my_layers)-1):
            init_layer = init_model.my_layers[i]
            this_layer = self.my_layers[i]
            weights, bias = init_layer.get_weights()
            weights = np.expand_dims(weights, 0)
            this_layer.set_weights((weights, bias))

    @staticmethod
    def binary_accuracy(y_true, y_pred, threshold=0.5):  # From original implementation
        """Calculate accuracy while ignoring -1s

        Parameters
        ----------
        y_true, y_pred : tensor
        threshold : float, optional
            How far from y_true should be counted as accurate, by default 0.5

        Returns
        -------
        tensor
        """
        if threshold != 0.5:
            threshold = K.cast(threshold, y_pred.dtype)
            y_pred = K.cast(y_pred > threshold, y_pred.dtype)
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    @staticmethod
    def mse_masked_loss(y_true, y_pred):
        """Calculate MSE while ignoring -1s

        Parameters
        ----------
            y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), K.floatx())
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return tf.losses.mse(y_true * mask, y_pred * mask)

    @staticmethod
    def binary_masked_loss(y_true, y_pred):  # From original implementation
        """Calculate binary crossentropy while ignoring -1s

        Parameters
        ----------
        y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), K.floatx())
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return tf.keras.losses.binary_crossentropy(y_true * mask, y_pred * mask)

    @staticmethod
    # From original implementation
    def sparse_categorical_masked_loss(y_true, y_pred):
        """Calculate categorical crossentropy while ignoring -1s

        Parameters
        ----------
        y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), K.floatx())
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)

    @staticmethod
    def masked_accuracy(y_true, y_pred):  # From original implementation
        """Calculate accuracy while ignoring -1s

        Parameters
        ----------
        y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), K.floatx())
        nb_mask = K.sum(K.cast(K.equal(y_true, -1), K.floatx()))
        nb_unmask = K.sum(mask)
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        ret = (K.sum(K.cast(K.equal(mask*y_true, K.round(mask*y_pred)),
               K.floatx()))-nb_mask)/nb_unmask
        return ret

    def call(self, inputs):
        """Call method used by tf.Model methods

        Parameters
        ----------
        inputs : tensor of self.input_shape            

        Returns
        -------
        tensor
            Results
        """
        x = inputs
        for layer in self.my_layers:
            x = layer(x)
        ret = []
        for layer in self.output_layers:
            ret.append(layer(x))
        return ret

    def show_scatter_analysis(self,
                              data,
                              relu=True,
                              normalize=True
                              ):
        """
        Show analysis graph on given data - dark points react strongly,
        light points react only a lightly.

        Arguments
        ---------
        data : np.array (n_points, point_dimensions)
          data points to be analysed
        relu : bool
          Should relu be applied to data before showing?
        normalize : bool
          Should data be normalized before showing?
        """
        analyzed = self.analyze_points(data, relu, normalize)
        plt.scatter(data[:, 0], data[:, 1], c=[str(v) for v in analyzed])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def analyze_points(self,
                       data,
                       relu=True,
                       normalize=True
                       ):
        """Return filter outputs for given weights

        Return outputs of the last filter layer, 
        which are used for the analysis.

        Parameters
        ----------
        data : np.array of shape (n cells, dimension)
            The cells wanted for analysis in a numpy array
        relu : bool, optional BUG:This doesn't work
            Should relu be applied to the outputs, by default True
        normalize : bool, optional BUG:This doesn't work
            Should the results be normalized into [0-1], by default True

        Returns
        -------
        np.array (n cells, n filters)
            Results for every cell in a np.array
        """
        sm = self.get_single_cell_model()
        return sm(data)

    def get_single_cell_model(self):
        """Generate model for single cell analysis.

        Returns
        -------
        SCellCNN
            Model for single cell analysis generated from this one.
        """
        return SCellCNN(self)

    def _select_top(self, tensor):
        """Return tensor containing mean of k largest numbers for given tensor"""
        return K.mean(tf.sort(tensor, axis=1)[:, -self.k:, :], axis=1)

    def get_config(self):
        """Return configuration for this model.

        Returns
        -------
        dict
        """
        return {"input_shape": self.my_input_shape,
                "classes": self.classes,
                "conv": self.conv,
                "k": self.k,
                "lr": self.lr,
                "activation": self.activation,
                "l1_weight": self.l1_weight,
                "dropout": self.dropout}

    def save(self, config_file, weights_file):
        """Save this model.

        Parameters
        ----------
        config_file : str or path
            Path to where the config .json file should be saved
        weights_file : str or path
            Path to where the weights .h5 file should be saved
        """
        json_config = self.get_config()
        with open(config_file, 'w') as file:
            json.dump(json_config, file)
        self.save_weights(weights_file)

    @staticmethod
    def load(config_file, weights_file=None):
        """Load a model from storage

        Parameters
        ----------
        config_file : str or path
            Path to the config .json file
        weights_file : str or path, optional
            Path to the weights .h5 file or None,
            if no weights are to be loaded,
            by default None

        Returns
        -------
        CellCNN
            Model of this class as was loaded.
        """
        with open(config_file, 'r') as file:
            config = json.load(file)
            model = CellCNN.load_from_dict(config)
        model.build(config["input_shape"])
        if weights_file:
            model.load_weights(weights_file)
        return model

    @staticmethod
    def load_from_dict(config):
        """Transled config dict to CellCNN model


        Used by CellCNN.load.

        Parameters
        ----------
        config : dict

        Returns
        -------
        CellCNN
            Model generated by the given dict
        """
        return CellCNN(input_shape=config["input_shape"],
                       classes=config["classes"],
                       conv=config["conv"],
                       k=config["k"],
                       lr=config["lr"],
                       activation=config["activation"],
                       l1_weight=config["l1_weight"],
                       dropout=config["dropout"])


class SCellCNN(CellCNN):
    """Model used for single cell analysis

    This is the model we want to use for analysis on single cell level,
    it is generated by a pre-trained CellCNN model and will return values
    output by the last filter layer.
    It skips all layers except the convolutional filter ones,
    and is not designed to be fitted/trained.
    """

    def __init__(self, original_model):
        """Initialize a single cell CellCNN

        Parameters
        ----------
        original_model : CellCNN
            Model on whose architecture and weights this one should be based.
        """
        super(CellCNN, self).__init__()
        self.my_layers = []
        self.classes = original_model.classes
        my_weights = []
        for layer in original_model.my_layers:
            if type(layer) == Conv1D:
                weights = layer.get_weights()[0][0]
                bias = layer.get_weights()[1]
                my_weights.append([weights, bias])
                self.my_layers.append(Dense(bias.shape[0], activation="relu"))
            # elif type(layer) == Dense:
            #     weights = layer.get_weights()[0]
            #     bias = layer.get_weights()[1]
            #     my_weights.append([weights, bias])
            #     self.my_layers.append(Dense(bias.shape[0],
            #                           activation=layer.activation))
            else:
                pass
        self.build(original_model.my_input_shape)
        for i in range(len(self.my_layers)):
            self.my_layers[i].set_weights(my_weights[i])

    def call(self, inputs):
        """Call method used by tf.Model methods."""
        x = inputs
        for i in self.my_layers:
            x = i(x)

        return x

    def show_importance(self, data, scale=False, filters=None, dimensions=(0, 1)):
        """Display values for cells returned by the last filter layer

        Parameters
        ----------
        data : np.array
            Cells for analysis
        scale : bool, optional
            Should the results be scaled before display,
            by default False
            BUG: If this is True, this will crash
        filters : list or None, optional
            For which filters should the output be displayed, by default None
        dimensions : tuple, optional
            Which 2 dimensions should work as the point location on graph,
            by default (0, 1)
        """
        def value_to_color(x):
            if (x >= 0.5):
                return [(x-0.5)*2, 0, 0]
            else:
                return[0, 0, 1-(2*x)]

        values = self.predict(data)
        if filters is None:
            filters = list(range(values.shape[-1]))
        for i in range(values.shape[-1]):
            if i in filters:
                current_values = values[:, i]
                if (scale):
                    print("MIN:", np.min(current_values))
                    print("MAX:", np.max(current_values))
                    current_values -= np.min(current_values)
                    current_values *= 1/np.max(current_values)
                fig = plt.scatter(data[:, dimensions[0]], data[:, dimensions[1]], c=[
                                  value_to_color(v) for v in current_values])
                plt.suptitle(f"Filter {i}, dimensions {dimensions}.")
                plt.show()


class InitCellCNN(CellCNN):
    def _build_layers(self):
        self.my_layers = []
        for i in range(len(self.conv)):
            if self.conv[i] != 0:
                self.my_layers.append(
                    Dense(self.conv[i],
                          activation=self.activation,
                          )
                )

    @staticmethod
    def load_from_dict(config, classes):
        return InitCellCNN(input_shape=(None, config["input_shape"][2]),
                           classes=classes,
                           conv=config["conv"],
                           k=config["k"],
                           lr=config["lr"],
                           activation=config["activation"])
