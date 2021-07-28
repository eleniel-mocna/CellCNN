import numpy as np
from InputData import InputData
from Dataset import DataDataset, DatasetSplit
from Models import CellCNN


def train_model(data,
                labels,
                multicell_size=1000,
                amount=5000,
                test_amount=1000,
                layers=[64, ],
                epochs=10):
    """Return a trained model for given arguments

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

    Returns
    -------
    CellCNN
        Trained CellCNN model
    """
    datasets, labels = Datasets_labels_from_data(data, labels)
    ID = InputData_from_Datasets(datasets, labels, multicell_size)
    return train_from_InputData(ID, amount, test_amount, layers, epochs)


def train_from_InputData(InputData,
                         amount=5000,
                         test_amount=1000,
                         layers=[64, ],
                         epochs=10):
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
                                  epochs=epochs)


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
    return Datasets_from_data(data), labels_from_labels_sheet(labels)


def labels_from_labels_sheet(sheet):
    return np.array(sheet)


def Datasets_from_data(data):
    ret = []
    for datum in data:
        ret.append(DataDataset(datum, shuffle=False))
    return ret


def InputData_from_Datasets(datasets,
                            labels,
                            multicell_size):
    return InputData(datasets, multicell_size, labels)


def data_labels_from_InputData(InputData,
                               amount=1000):
    return InputData.get_multi_cell_inputs(amount)


def train_from_data_labels(data,
                           labels,
                           test_data,
                           test_labels,
                           layers=[64, ],
                           epochs=10):
    data = np.array(data)
    labels = np.array(labels)
    input_shape = list(data.shape)
    input_shape.insert(0, None)
    input_shape = tuple(input_shape)
    model = CellCNN(input_shape=input_shape, conv=layers)
    model.fit(data, labels, validation_data=(
        test_data, test_labels), epochs=epochs)
    return model
if __name__ == "__main__":
    d1 = np.random.random((5132,2))
    d2 = np.random.random((5456,2))+1
    labels = ((0,),(1,))
    model = train_model((d1, d2), labels, 100, 1000, 500)
    model.get_single_cell_model().show_importance(np.random.random((1000, 2)))