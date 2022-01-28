import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Lambda, Layer, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

#TODO: Bugfix masking.


class L1Layer(Layer):
    """Layer which passes values through and applies
    1L regularization to them.
    """

    def __init__(self, loss_weight=1e-10, use_vector=False):
        """Initialize the layer.

        Parameters
        ----------
        loss_weight : float, optional
            How strong should the regularization be, by default 1e-10
        use_vector : bool, optional
            Should the l1 regu be applied to every filter with a different strength?
            if True: l1 strength for i-th vector is:
                loss_weight * (2**i)
        """
        super(L1Layer, self).__init__()
        self.loss_weight = loss_weight
        self.use_vector = use_vector

    def build(self, input_shape):       
        # self.loss_vector = tf.constant((-10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10), dtype="float32")    
        super(L1Layer, self).build(input_shape)
        self.loss_vector = tf.range(0,input_shape[-1], dtype="float32")    
        self.loss_vector = 2**self.loss_vector
        self.loss_vector *= self.loss_weight
        if self.use_vector:
            self.loss_multiplier = self.loss_vector
        else:
            self.loss_multiplier = self.loss_weight
    def compute_output_shape(self,input_shape):
        return input_shape
    def call(self, inputs):
        filter_responses = tf.reduce_sum(inputs,(0,1))
        loss_results = self.loss_multiplier*filter_responses
        self.add_loss(tf.reduce_sum(loss_results))
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
                 l1_weight=5e-20,
                 l1_vector_like = False,
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
        l1_vector_like : bool
            Should the l1 regu be applied to every filter with a different strength?
            if True: l1 strength for i-th vector is:
                loss_weight * (2**i)
        dropout : float, optional
            Strength of dropout before every filter layer,
            by default 0.25
        """
        super(CellCNN, self).__init__()
        self.my_input_shape = tuple(input_shape)
        if type(classes) == int:
            self.classes = tuple([classes])
        else:
            self.classes = tuple(classes)
        if type(conv) == int:
            self.conv = tuple([conv])
        else:
            self.conv = tuple(conv)
        self.k = k
        self.lr = lr
        self.activation = activation
        self.l1_weight = l1_weight
        self.l1_vector_like = l1_vector_like
        self.dropout = dropout
        self._build_layers()
        self._compile()
        self.build(tuple(self.my_input_shape))
        self.train_acc_metric = keras.metrics.BinaryAccuracy()
        self.val_acc_metric = keras.metrics.BinaryAccuracy()

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
                           activation=self.activation#,
                        #    kernel_regularizer="l2"
                           )
                )

        self.my_layers.append(L1Layer(self.l1_weight, self.l1_vector_like))
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
                # self.loss_functions.append(
                #    tf.keras.losses.SparseCategoricalCrossentropy())
                self.loss_functions.append(
                    CellCNN.sparse_categorical_masked_loss)
            elif i < 1:
                self.output_layers.append(
                    Dense(-i, activation="softmax", name=layer_name))
                print("W: Earth mover loss seems not to work. Be careful!", file=sys.stderr)
                # self.loss_functions.append(
                #    tf.keras.losses.SparseCategoricalCrossentropy())
                self.loss_functions.append(
                    CellCNN.earth_mover_loss)
            else:
                raise ValueError(f"Invalid output layer specification given: {i}!")
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
                    batch_size=256):
        """Initialize random weights with better than random values

        This just splits given data into random labels and tries to fit
        to them. This should make the training more stable

        Parameters
        ----------
        data : np.array of shape (n inputs, n cells, n dims)
            Data to which the model should train better.
        labels : np.array of shape (n inputs*ncells,1), optional
            Predefined labels, by default None
        n_classes : int, optional
            To how many classes should/is the data (be) split, by default 10
        epochs : int, optional, by default 10
        batch_size : int, optional, by default 256
        """
        init_model = InitCellCNN.load_from_dict(self.get_config(), n_classes)
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
        if labels is None:
            labels = np.random.randint(
                low=n_classes, size=data.shape[0], dtype='l')  # Does this work?

        conv_layer_indices = []
        for i in range(len(self.my_layers)):
            if type(self.my_layers[i]) == Conv1D:
                conv_layer_indices.append(i)

        init_model.fit(data, labels, batch_size=batch_size, epochs=epochs)
        for i in range(len(init_model.my_layers)-1):
            init_layer = init_model.my_layers[i]
            this_layer = self.my_layers[conv_layer_indices[i]]
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
    def earth_mover_loss(y_true, y_pred):
        """Calculate the wasserstein metric, while ignoring -1s.

        Parameters
        ----------
        y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), "bool")
        
        # In graph code above function yields different dimensions than in eager.
        # This squeeze ensures mask is compatible in both execution forms.
        if len(mask.shape)>1:
            mask = tf.squeeze(mask,1)
        y_true_masked = tf.boolean_mask(y_true,mask)
        y_pred_masked = tf.boolean_mask(y_pred,mask)
        cdf_true = K.cumsum(tf.one_hot(tf.cast(y_true_masked, "int32"), y_pred.shape[1]), axis=-1)
        cdf_pred = K.cumsum(y_pred_masked, axis=-1)
        emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
        return emd

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
        mask = K.cast(K.not_equal(y_true, -1), "bool")
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        ret = tf.losses.mse(y_true_masked, y_pred_masked)
        #Replace nan values with 0s
        return tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)
        

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
        mask = K.cast(K.not_equal(y_true, -1), "bool")
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        ret = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
        #Replace nan values with 0s
        return tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

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
        # return(tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred))
        mask = K.cast(K.not_equal(y_true, -1), "bool")
        
        # In graph code above function yields different dimensions than in eager.
        # This squeeze ensures mask is compatible in both execution forms.
        if len(mask.shape)>1:
            mask = tf.squeeze(mask,1)
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        y_true_masked = tf.boolean_mask(y_true,mask)
        y_pred_masked = tf.boolean_mask(y_pred,mask)
        ret = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
        #Replace nan values with 0s
        return tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

    @staticmethod
    def masked_accuracy(y_true, y_pred):  # From original implementation
        """Calculate accuracy while ignoring -1s on binary classification problems

        Note
        ----
        For binary or regression problems this metric doesn't make any sence...

        Parameters
        ----------
        y_true, y_pred : tensor

        Returns
        -------
        tensor
        """
        mask = K.cast(K.not_equal(y_true, -1), "bool")
        if len(mask.shape)>1:
            mask = tf.squeeze(mask,1)
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        y_true_masked = tf.boolean_mask(y_true,mask)
        y_pred_masked = tf.boolean_mask(y_pred,mask)
        if len(y_pred.shape)==1: # This is a binary accuracy
            y_true_masked = tf.concat(y_true_masked, 1-y_true_masked)
            y_pred_masked = tf.concat(y_pred_masked, 1-y_pred_masked)
        if len(y_pred.shape)==2: # This is a categorical accuracy
            tf.one_hot(tf.cast(y_true_masked, "int32"), y_pred_masked.shape[1])

        return tf.keras.metrics.sparse_categorical_accuracy(y_true_masked, y_pred_masked)
        # mask = K.cast(K.not_equal(y_true, -1), K.floatx())
        # nb_mask = K.sum(K.cast(K.equal(y_true, -1), K.floatx()))
        # nb_unmask = K.sum(mask)
        # y_true = K.cast(y_true, K.floatx())
        # y_pred = K.cast(y_pred, K.floatx())
        # ret = (K.sum(K.cast(K.equal(mask*y_true, K.round(mask*y_pred)),
        #        K.floatx()))-nb_mask)/nb_unmask
        # return ret

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
    
    # @tf.function TODO: Uncomment this
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        my_loss_values = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)  # Forward pass
            if len(self.loss) > 1:
                for i in range(len(self.loss)):
                    my_loss = self.loss[i]
                    my_loss_values.append(tf.math.reduce_sum(my_loss(y[i], y_pred[i])))
                # loss = tf.math.reduce_sum(my_loss_values) + tf.math.reduce_sum(self.losses)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            else:
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            regu_loss = tf.math.reduce_sum(self.losses)
            

            # Update metrics (includes the metric that tracks the loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1e5, 1e5)) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(capped_gvs, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        ret = {m.name: m.result() for m in self.metrics}
        ret["loss"] = loss
        ret["regu_loss"] = regu_loss
        if my_loss_values:
            for i in range(len(self.loss)):
                grads = tape.gradient(my_loss_values[i],trainable_vars)
                
                # Remove Nones (in output layers not corresponding to this
                # output.)
                clean_grads = []
                for grad in grads:
                    if grad != None:
                        clean_grads.append(tf.norm(grad))

                # Calculate average gradient size, from core layers
                # Last 2 gradients describe gradients to output layer's
                # weights and bias.
                my_grad = tf.math.reduce_mean(clean_grads[:-2]) 
                if my_grad == None:
                    my_grad = 0
                ret[f"gradient_{i}"] = my_grad
        grads = tape.gradient(regu_loss,trainable_vars)
        clean_grads = []
        for grad in grads:
            if grad != None:
                clean_grads.append(tf.norm(grad))
        my_grad = tf.math.reduce_mean(clean_grads)
        ret["regu_grad"] = my_grad
        return ret
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        my_loss_values = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)  # Forward pass
            if len(self.loss) > 1:
                for i in range(len(self.loss)):
                    my_loss = self.loss[i]
                    my_loss_values.append(tf.math.reduce_sum(my_loss(y[i], y_pred[i])))
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                # loss = tf.math.reduce_sum(my_loss_values) + tf.math.reduce_sum(self.losses)
            else:
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            regu_loss = tf.math.reduce_sum(self.losses)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        ret = {m.name: m.result() for m in self.metrics}
        ret["loss"] = loss
        ret["regu_loss"] = regu_loss
        trainable_vars = self.trainable_variables
        if my_loss_values:
            for i in range(len(self.loss)):
                grads = tape.gradient(my_loss_values[i],trainable_vars)
                
                # Remove Nones (in output layers not corresponding to this
                # output.)
                clean_grads = []
                for grad in grads:
                    if grad != None:
                        clean_grads.append(tf.norm(grad))

                # Calculate average gradient size, from core layers
                # Last 2 gradients describe gradients to output layer's
                # weights and bias.
                my_grad = tf.math.reduce_mean(clean_grads[:-2]) 
                if my_grad == None:
                    my_grad = 0
                ret[f"gradient_{i}"] = my_grad
        grads = tape.gradient(regu_loss,trainable_vars)
        clean_grads = []
        for grad in grads:
            if grad != None:
                clean_grads.append(tf.norm(grad))
        my_grad = tf.math.reduce_mean(clean_grads)
        ret["regu_grad"] = my_grad
        return ret



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
        self.loss_functions = [tf.keras.losses.SparseCategoricalCrossentropy()]
        self.output_layers = [Dense(self.classes[0], activation="softmax")]

    @staticmethod
    def load_from_dict(config, classes):
        return InitCellCNN(input_shape=(None, config["input_shape"][2]),
                           classes=classes,
                           conv=config["conv"],
                           k=config["k"],
                           lr=config["lr"],
                           activation=config["activation"])
