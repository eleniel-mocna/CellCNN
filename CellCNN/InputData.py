
import numpy as np
import matplotlib.pyplot as plt
from Dataset import DatasetSplit, Dataset

RNG = np.random.default_rng(211)


class InputData:
    """Class for storage and output of data from Dataset objects.

    To extract data points from this class, use get_multi_cell_inputs().

    Arguments
    ---------
    datasets : list/tuple of datasets
        Datasets, from which the data is taken.
    multi_cell_size : int
        How many values should be in one multi-cell input.
    labels : 2D numpy array 
        Labels for individual datasets (or their numbers when empty)
    """

    def __init__(
            self,
            datasets,
            multi_cell_size=1000,
            labels=None):

        if labels is None:
            self.labels = np.arange(len(datasets))
            self.labels = np.expand_dims(self.labels, -1)
        else:
            self.labels = labels

        self.datasets = datasets
        self.multi_cell_size = multi_cell_size
        self._check_datasets()
        self.dimension = self.datasets[0].dimension

    def _check_datasets(self):
        dimension = self.datasets[0].dimension
        for i in self.datasets:
            assert (i.dimension == dimension)

    @property
    def length(self):
        """Get the number of datasets stored in this object

        Returns
        -------
        int
        """
        return len(self.datasets)

    def get_multi_cell_inputs(self, amount, split_type=DatasetSplit.TRAIN):
        """
        Arguments
        ---------
        amount : int
            How many samples will be returned
        split_type : DatasetSplit enum
            From which part of the dataset should the data be taken
            BUG: This argument is ignored!
        
        Returns
        -------
        multi_cell_inputs : np.array(`amount`, multi_cell_size, dimension)
            Data (X).
        multi_cell_labels : list(n_labels, `amount`)
            Labels (Y).
        """
        retX = np.ndarray((amount, self.multi_cell_size, self.dimension))
        retY = np.ndarray((amount, self.labels[0].shape[0]))
        indices = RNG.integers(0, self.length, size=(amount), dtype="int8")
        for i in range(amount):
            retX[i] = self.datasets[indices[i]].get_multi_cell_input(
                self.multi_cell_size)
            retY[i] = self.labels[indices[i]]
        retY = list(retY.transpose())
        return (retX, retY)

    def plot(self, amount=5000,
             color_values=["#ff2222", "#22ff22", "#2222ff",
                           "#ff22ff", "#ffff22", "#22ffff"],
             dataset_split=DatasetSplit.TRAIN):
        DISPERSION = 10
        fig = plt.figure(figsize=(8, 6), dpi=80)
        n_scatters = len(self.datasets)*DISPERSION
        i_list = [i % len(self.datasets) for i in range(n_scatters)]
        for i in i_list:
            data = self.datasets[i].get_multi_cell_input(
                amount//DISPERSION, dataset_split)
            color = color_values[i % len(color_values)]
            plt.scatter(data[:, 0], data[:, 1], color=color, s=5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


class ExampleInputData(InputData):
    def __init__(self,
                 amounts=10000,
                 multi_cell_size=1000):
        """Example data for CellCNN analysis
        Data is a circle with a 

        Parameters
        ----------
        amounts : int, optional
            [description], by default 10000
        multi_cell_size : int, optional
            [description], by default 1000

        Returns
        -------
        [type]
            [description]
        """
        negative = Dataset(amounts)
        positive = Dataset(amounts,
                           center_density=0.15)
        return super().__init__((negative, positive), multi_cell_size)


if __name__ == "__main__":
    ExampleInputData()
