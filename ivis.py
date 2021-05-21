import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  InputLayer, Dense, Flatten, Reshape
from tensorflow.keras import backend as K
from tensorflow.python.keras import regularizers
tf.config.run_functions_eagerly(False)
tf.keras.backend.set_floatx('float64')
class Ivis(Model):
    """
    Ivis model for dimensional reduction on MNIST data.
    """
    def __init__(self,
                 verbose=True,
                 encoder=True,
                 recoder=True,
                 ivis_weight=5,
                 recoder_weight=1):                 
        super(Ivis, self).__init__()
        if (type(self) == Ivis):
            assert encoder and recoder, "Base Ivis model needs to have both decoder and encoder!"
        self.verbose = verbose
        self.ivis_weight = ivis_weight
        self.recoder_weight=recoder_weight
        self._build_layers(encoder, recoder)
        self.compile(
            optimizer="adam",
            run_eagerly=None)
    def _build_layers(self, encoder, recoder):
        """Prepare model layers.

        Parameters
        ----------
        encoder : bool
            Should encoder be generated?
        recoder : bool
            Should decoder be generated?
        """
        if (encoder):
            self.encoder_layers = [
                InputLayer((28, 28, 1)),
                Flatten(),
                Dense(64,activation="relu"),
                Dense(64,activation="relu"),
                Dense(2)
                ]
        if (recoder):
            self.recoder_layers = [
                InputLayer(2),
                Dense(128, activation="relu"),
                Dense(512, activation="relu"),
                Dense(784, activation="sigmoid"),
                Reshape((28,28))
                ]

    def call(self, inputs):
        out_0 = self.encoder_out(inputs[0])
        out_1 = self.encoder_out(inputs[1])
        out_2 = self.encoder_out(inputs[2])
        recoded = self.recoder_out(out_0)
        ivis_loss = (Ivis.triplet(out_0,
                               out_1,
                               out_2,
                               distance=Ivis.distance,))
        
        recoder_loss = (Ivis.recoder_loss(inputs[0],
                                  recoded,
                                  distance=Ivis.distance_s))
        self.add_loss(self.ivis_weight * ivis_loss
                    + self.recoder_weight * recoder_loss)
        return out_0, out_1, out_2, recoded

    def encoder_out(self,
                    input_layer):
        """
        Call through encoder layers.
        """
        
        x = input_layer
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    def recoder_out(self,
                    input_layer):
        """
        Call through recoder layers.
        """
        x = input_layer
        for layer in self.recoder_layers:
            x = layer(x)
        return x

    @staticmethod
    def distance(x, y):
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    @staticmethod
    def distance_s(x, y):
        return tf.reduce_sum((tf.reduce_sum(K.square(x - y), axis=-2)),axis=-1, keepdims=True)
    @staticmethod
    def triplet(anchor,
                positive,
                negative,
                distance=None):
        """
        Return loss for ivis encoder.
        """
        if (distance == None): distance = Ivis.distance
        anchor_positive_distance = distance(anchor, positive)
        anchor_negative_distance = distance(anchor, negative)
        positive_negative_distance = distance(positive, negative)
        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)
        return K.mean(K.maximum(anchor_positive_distance-minimum_distance+10.,0))

    @staticmethod
    def recoder_loss(original,
                     recoded,
                     distance = None):
        if (distance == None): distance = Ivis.distance_s
        return distance((original[:,:,:,0]), (recoded))

    def get_encoder(self):
        return IvisEncoder(self)
    def get_recoder(self):
        return IvisRecoder(self)

class IvisEncoder(Ivis):
    """
    Model made from ivis model -> only encodes images.
    Input as into original ivis model. (None, 28, 28)
    Output 2D dimensionaly reduced data. (None, 2)
    """
    def __init__(self, original_model):
        super(IvisEncoder, self).__init__(encoder=True,
                                        recoder=False)
        self.build((None, 28, 28, 1)) # Cannot set weights before the model is built.    
        for i in range(len(original_model.encoder_layers)):
            self.encoder_layers[i].set_weights(original_model.encoder_layers[i].get_weights())
    def call(self, inputs):
        return self.encoder_out(inputs)

class IvisRecoder(Ivis):
    """
    Model made from ivis model -> only recodes from 2D data.
    Input is 2D vector. (None, 2)
    Output is image as original input into ivis (None, 28, 28)
    """
    def __init__(self, original_model):
        super(IvisRecoder, self).__init__(encoder=False,
                                        recoder=True)
        self.build((None,2))
        for i in range(len(original_model.recoder_layers)):
            self.recoder_layers[i].set_weights(original_model.recoder_layers[i].get_weights())
    def call(self, inputs):
        return self.recoder_out(inputs)
