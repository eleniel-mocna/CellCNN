import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

#TODO: Bugfix masking.

class CellCNN(Model):
    def __init__(self,
                 input_shape,
                 n_classes = 2,
                 conv = [16,],
                 k=25,
                 lr=0.01,
                 activation="relu"):
        """
        Build CellCNN model

        Arguments
        ---------
            conv : list of ints
                list of filter sizes of convolutional layers
            k : int
                number of pooled convolutional outputs after last conv layer
        """
        super(CellCNN, self).__init__()
        self.my_input_shape=input_shape
        self.n_classes = n_classes
        self.conv = conv
        self.k = k
        self.lr = lr
        self.activation = activation
        self._build_layers()
        self._compile()            
        self.build(tuple(self.my_input_shape))
    def _build_layers(self):
        """Build layers for arguments given in __init__()
        """
        self.my_layers = []
        for i in range(len(self.conv)):
            if self.conv[i] != 0:
                self.my_layers.append(
                                    Conv1D(filters=self.conv[i],
                                        kernel_size=1,
                                        activation=self.activation,
                                        )
                                    )            
        self.my_layers.append(Lambda(self._select_top,
                                    output_shape=(1,),
                                    name="pooling"
                                    )
                            )
    def _compile(self):
        assert self.n_classes > 1, "There must be at least 2 classes for classifier!"
        if self.n_classes == 2:
            self.my_layers.append(Dense(1, activation="sigmoid"))
            loss_fn = tf.keras.losses.binary_crossentropy
            # loss_fn = CellCNN.binary_masked_loss
        else:
            self.my_layers.append(Dense(self.n_classes, activation="linear"))
            loss_fn = tf.keras.losses.mean_squared_error
            # loss_fn = CellCNN.sparse_categorical_masked_loss
        self.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss=loss_fn,
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
                    batch_size=256):
        init_model = InitCellCNN.load_from_dict(self.get_config(), n_classes)
        if labels is None: labels = np.random.randint(low=n_classes,size=data.shape[0], dtype='l') # Does this work?

        init_model.fit(data, labels, batch_size=batch_size, epochs=epochs)
        for i in range(len(init_model.my_layers)-1):
            init_layer = init_model.my_layers[i]            
            this_layer = self.my_layers[i]
            weights, bias = init_layer.get_weights()
            weights = np.expand_dims(weights, 0)
            this_layer.set_weights((weights, bias))

    @staticmethod
    def binary_accuracy(y_true, y_pred, threshold=0.5): # From original implementation
        if threshold != 0.5:
            threshold = K.cast(threshold, y_pred.dtype)
            y_pred = K.cast(y_pred > threshold, y_pred.dtype)
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)    
    @staticmethod
    def binary_masked_loss(y_true, y_pred): # From original implementation
        mask = K.cast(K.not_equal(y_true,-1), K.floatx())
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return tf.keras.losses.binary_crossentropy(y_true * mask, y_pred * mask)
    @staticmethod
    def sparse_categorical_masked_loss(y_true, y_pred): # From original implementation
        mask = K.cast(K.not_equal(y_true,-1), K.floatx())
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)
    @staticmethod
    def masked_accuracy(y_true, y_pred): # From original implementation
        mask=K.cast(K.not_equal(y_true,-1),K.floatx())
        nb_mask=K.sum(K.cast(K.equal(y_true,-1),K.floatx()))
        nb_unmask=K.sum(mask)
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        ret= (K.sum(K.cast(K.equal(mask*y_true, K.round(mask*y_pred)),K.floatx()))-nb_mask)/nb_unmask
        return ret    
    def call(self, inputs):
        x = inputs
        for layer in self.my_layers:
            x = layer(x)      
        return x        
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
        plt.scatter(data[:,0], data[:,1], c=[str(v) for v in analyzed])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    def analyze_points(self,
                       data,
                       relu=True,
                       normalize=True
                       ):
        """Return np.array of values out of trained filters on given data."""
        print(DeprecationWarning("Use SCellCNN model instead."))
        x = data
        for layer in self.my_layers:
            if type(layer) == Conv1D:
                x = (np.matmul(x,
                  np.array(layer.weights[0][0]))
                  + np.array(layer.weights[1]))
        ret = x                  
        if relu:
            ret = self._relu(ret)
        if normalize:
            ret = self._normalize_output_values(ret)
        return ret
    def get_single_cell_model(self):
        return SCellCNN(self)      
    def _relu(self, values):
        """ Aply RELU to given values"""
        assert len(values.shape)==2, "Non-standard input into RELU!"
        for i in range(values.shape[0]):
          for j in range(values.shape[1]):
            values[i,j] = max(0, values[i,j])
        return values
    def _normalize_output_values(self, values):
        """
        Normalize given values for plt.
        Reduce dimension by last dimension and retun normalized sums
        """
        assert len(values.shape)==2, "Non-standard input into normalization!"
        values = np.sum(values, -1)
        values -= np.min(values)
        values/= np.max(values)
        return values 
    def _select_top(self, tensor,):
        """Return tensor containing mean of k largest numbers for given tensor"""
        return K.mean(tf.sort(tensor, axis=1)[:, -self.k:, :], axis=1)
    def get_config(self):
        return {"input_shape" : self.my_input_shape,
                "n_classes": self.n_classes,
                "conv": self.conv,
                "k": self.k,
                "lr": self.lr,
                "activation" :self.activation}
    def save(self, config_file, weights_file):
        json_config = self.get_config()
        with open(config_file, 'w') as file:
            json.dump(json_config, file)
        self.save_weights(weights_file)
    @staticmethod
    def load(config_file, weights_file=None):
        with open(config_file, 'r') as file:
            config = json.load(file)
            model = CellCNN.load_from_dict(config)
        model.build(config["input_shape"])
        if weights_file:
            model.load_weights(weights_file)
        return model
    @staticmethod
    def load_from_dict(config):
        return CellCNN(input_shape=config["input_shape"],
                            n_classes=config["n_classes"],
                            conv=config["conv"],
                            k=config["k"],
                            lr=config["lr"],
                            activation=config["activation"])
class SCellCNN(CellCNN):
    def __init__(self, original_model):
        super(CellCNN, self).__init__()
        self.my_layers = []
        self.n_classes = original_model.n_classes
        my_weights = []
        for layer in original_model.layers:
            if type(layer) == Conv1D:
                weights = layer.get_weights()[0][0]
                bias = layer.get_weights()[1]
                my_weights.append([weights, bias])
                self.my_layers.append(Dense(bias.shape[0], activation="relu"))
            elif type(layer) == Dense:                
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                my_weights.append([weights, bias])
                self.my_layers.append(Dense(bias.shape[0],
                                      activation=layer.activation))
            else:
                pass
        self.build((None,original_model.my_input_shape[-1]))
        for i in range(len(self.my_layers)):
            self.my_layers[i].set_weights(my_weights[i])
    def call(self,inputs):
        x = inputs
        for i in self.my_layers:
            x = i(x)
            
        return x
    def show_importance(self, data, scale=False, filters=None, dimensions=(0,1)):
        def value_to_color(x):
            if (x >= 0.5):
                return [(x-0.5)*2,0,0]
            else:
                return[0,0,1-(2*x)]

        values = self.predict(data)
        if filters is None:
            filters = list(range(values.shape[-1]))
        for i in range(values.shape[-1]):
            if i in filters:
                current_values = values[:,i]
                if (scale):
                    print("MIN:", np.min(current_values))
                    print("MAX:", np.max(current_values))
                    current_values -= np.min(current_values)
                    current_values *= 1/np.max(current_values)
                fig = plt.scatter(data[:,dimensions[0]], data[:,dimensions[1]], c=[value_to_color(v) for v in current_values])
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
    def load_from_dict(config, n_classes):
        return InitCellCNN(input_shape=(None, config["input_shape"][2]),
                            n_classes=n_classes,
                            conv=config["conv"],
                            k=config["k"],
                            lr=config["lr"],
                            activation=config["activation"])
    
