import numpy as np
import tensorflow as tf
from InputData import InputData
from Dataset import DataDataset, DatasetSplit
from Models import CellCNN


def train_model(data,
                labels,
                multicell_size=1000,
                amount=5000,
                test_amount=1000,
                layers=[64, ],
                epochs=10,
                classes=[2, ],
                k=25,
                lr=1e-7,
                activation="relu",
                l1_weight=0.01,
                dropout=0.25,
                patience=3):    
    """Train a model for given arguments

    Parameters
    ----------
    data : list-like
        List of shape (n data points, n cells, n parameters per cell),
        describing data
    labels : array-like
        List of shape (n data points, n labels)
        First dimension must be sorted according to `data`
    multicell_size : int, optional
        N of cells in one multicell input, by default 1000
    amount : int, optional
        N of generated multicell inputs, by default 5000
    test_amount : int, optional
        Amount of generated multicell inputs used as test dataset, by default 1000
    layers : list, optional
        Number and shape of layers in model, by default [64, ]
    epochs : int, optional
        N of epochs, by default 10
    classes : list, optional
        Description of labels used, where:
            0: linear variable,
            2: binary classification,
            n > 2: n-nary classification.
        E.g. [2,0] expects labels to express a binary classification
        problem and a regression problem.
        By default [2, ].
    k : int, optional
        Number of cells that go through pooling
        after the last filter layer, 
        by default 25.
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
        by default 0.25.
    patience : int, optional
        How many epochs where val_loss is increasing should we wait
        before we stop the training, by default 3.
    Returns
    -------
    CellCNN
        Trained CellCNN model
    """
    datasets, labels = Datasets_labels_from_data(data, labels)
    ID = InputData(datasets, multicell_size, labels)
    return train_from_InputData(ID, amount, test_amount,
                                layers=layers,
                                epochs=epochs,
                                classes=classes,
                                k=k,
                                lr=lr,
                                activation=activation,
                                l1_weight=l1_weight,
                                dropout=dropout,
                                patience=patience)


def train_from_InputData(InputData,
                         amount=5000,
                         test_amount=1000,
                         layers=[64, ],
                         epochs=10,
                         classes=[2, ],
                         k=25,
                         lr=0.01,
                         activation="relu",
                         l1_weight=0.01,
                         dropout=0.25,
                         patience=3):
    """Return trained CellCNN model to given arguments

    Parameters
    ----------
    InputData : InputData
        InputData object containing data and labels
    amount : int, optional
        Amount of generated multicell inputs, by default 5000
    test_amount : int, optional
        Amount of generated multicell inputs used as test dataset, by default 1000
    layers : list, optional
        Number and shape of layers in model, by default [64, ]
    epochs : int, optional
        N of epochs, by default 10

    Returns
    -------
    CellCNN
        Trained CellCNN model
    """
    data_train, labels_train = InputData.get_multi_cell_inputs(
        amount, split_type=DatasetSplit.TRAIN)
    data_test, labels_test = InputData.get_multi_cell_inputs(
        test_amount, split_type=DatasetSplit.TEST)
    return train_from_data_labels(data_train,
                                  labels_train,
                                  test_data=data_test,
                                  test_labels=labels_test,
                                  layers=layers,
                                  epochs=epochs,
                                  classes=classes,
                                  k=k,
                                  lr=lr,
                                  activation=activation,
                                  l1_weight=l1_weight,
                                  dropout=dropout,
                                  patience=patience)


def Datasets_labels_from_data(data,
                              labels):
    """Return list of datasets and np.array of labels

    Parameters
    ----------
    data : list-like
        List of shape (n data points, n cells, n parameters per cell),
        describing data
    labels : array-like
        List of shape (n data points, n labels)
        First dimension must be sorted according to `data`

    Returns
    -------
    (list, np.array)
        Datasets and labels ready for `InputData` class
    """
    return Datasets_from_data(data), np.array(labels)

def Datasets_from_data(data):
    """Convert cell readings to Dataset object

    Parameters
    ----------
    data : list-like of shape (n data points, n cells, n parameters per cell)
        Input data

    Returns
    -------
    list
        Datasets created from given data.
    """
    ret = []
    for datum in data:
        ret.append(DataDataset(datum, shuffle=False))
    return ret

def train_from_data_labels(data,
                           labels,
                           test_data,
                           test_labels,
                           layers=[64, ],
                           epochs=10,
                           classes=[2, ],
                           k=25,
                           lr=0.01,
                           activation="relu",
                           l1_weight=0.01,
                           dropout=0.25,
                           patience=3):
    """Train model based on given data

    Parameters
    ----------
    data : np.array of shape (n multicells, n cells, dimension)
        Training data
    labels : np.array of shape (n multicells, n labels)
        Labels for training data
    test_data : np.array of shape (n multicells, n cells, dimension)
        Data for testing
    test_labels : np.array of shape (n multicells, n labels)
        Labels for test data
    layers : list, optional
        Layer definition passed to CellCNN model, by default [64, ]
    epochs : int, optional, by default 10        
    classes : list, optional
        Labels description given to CellCNN model, by default [2, ]
    k : int, optional
        Pooling setting given to CellCNN model, by default 25
    lr : float, optional
        Learning rate, by default 0.01
    activation : str, optional
        Activation function, by default "relu"
    l1_weight : float, optional, by default 0.01
    dropout : float, optional
        Dropout density, by default 0.25
    patience : int, optional
        How many epochs where val_loss is increasing should we wait
        before we stop the training, by default 3.

    Returns
    -------
    CellCNN model
        Trained model
    """
    input_shape = list(data.shape)
    input_shape[0] = None
    input_shape = tuple(input_shape)
    model = CellCNN(input_shape=input_shape, conv=layers, classes=classes,
                    k=k, lr=lr, activation=activation, l1_weight=l1_weight, dropout=dropout)
    model.init_random(data, epochs=1)
    model.fit(data, labels, validation_data=(
        test_data, test_labels), epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        batch_size = 128)
    return model


if __name__ == "__main__":
    d1 = np.random.random((5132, 2))
    d2 = np.random.random((5456, 2))+1
    labels = ((0,), (1,))
    model = train_model((d1, d2), labels, 100, 1000, 500)
    model.get_single_cell_model().show_importance(np.random.random((1000, 2)))
