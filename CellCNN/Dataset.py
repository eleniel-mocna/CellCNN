import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

RNG = np.random.default_rng(211)


class DatasetSplit(Enum):
    """Enum defining which part of the dataset should be used"""
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class DimensionError(ValueError):
    """This error is thrown when you are trying to do something
    in more (or less) dimensions then the something is designed for...
    """
    pass


def plot_datasets(list_of_datasets, list_of_styles=("g.", "r.", "b.")):
    """Deprecated. Will be removed soon."""
    print("WARNING: I don't believe plot_datasets function works well...")
    print("WARNING: Also, it is deprecated")
    plt.figure(figsize=(10, 10), dpi=80)
    for i in range(len(list_of_datasets)):
      plt.plot(list_of_datasets[i][:, 0], list_of_datasets[i][:, 1],
               list_of_styles[i % len(list_of_styles)],
               alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
# TODO: Make an abstract class for dataset and these "example datasets" just inherit from it.


class Dataset:
    """
    Generate an elipse dataset for learning with ratio of
    `center_density` points in the middle from total `amount` points

    Arguments
    ---------
    amount : int
        number of points generated
    dimension : int
        dimension of data
    center_density : float [0,1]
        ratio of how many points are in the center
    radius : float
        mean radius of outer ring
    center_scatter : float
        radius of the center circle
    outer_scatter : float
        width of the outer ring
    center_radius : float
        radius of center ring
    test_split : float
        fraction of data saved for testing
    validation_split : float
        fraction of data saved for validation
    """

    def __init__(self,
                 amount,
                 dimension=2,
                 center_density=0.1,
                 radius=1,
                 center_scatter=0.5,
                 center_radius=0.1,
                 outer_scatter=0.3,
                 test_split=0.2,
                 validation_split=0.2,
                 offsetX=1,
                 offsetY=1
                 ):
        self.amount = amount
        self.dimension = dimension
        self.center_density = center_density
        self.radius = radius
        self.center_scatter = center_scatter
        self.center_radius = center_radius
        self.outer_scatter = outer_scatter
        self.offsetX = offsetX
        self.offsetY = offsetY

        self._generate_data()
        self.train_data, self.test_data, self.validation_data = self._split_data(
            test_split, validation_split)

    def plot_data(self, style='b.', verbose=1):
        """Plot data saved in this dataset

        Parameters
        ----------
        style : str, optional
            plot style given to plt.plot, by default 'b.'
        verbose : int, optional
            0: doesn't do anything when more then 2 dimensions given,
            1: prints a dimension warning,
            2: throws a DimensionError exception,
            by default 1

        Raises
        ------
        DimensionError
            When more than 2 dimensions given and verbose==2.
        """
        if self.dimension != 2:
            if verbose == 2:
                raise DimensionError(
                    f"Due to your screen being only 2D, unable to plot {self.dimension}D data! Try changing `verbose`")
            if verbose == 1:
                print("WARNING: Plotting only 2 first dimensions!")
        plt.plot(self.data[:, 0], self.data[:, 1], style)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def get_multi_cell_input(self, size, split=DatasetSplit.TRAIN):
        """Randomly choose `size` of values from dataset. 

        TODO: Make this not to be random choice but iterate through the values.

        Parameters
        ----------
        size : int
            How many cells should be taken
        split : DatasetSplit enum, optional
            From which part of the dataset do you want the data to be,
            by default DatasetSplit.TRAIN

        Returns
        -------
        np.array of shape (size, dimensions)
            Returned cells in a np.array.

        Raises
        ------
        ValueError
            When incorrect split is given.
        """
        if split == DatasetSplit.TRAIN:
            return RNG.choice(self.train_data, size=size)
        elif split == DatasetSplit.TEST:
            return RNG.choice(self.test_data, size=size)
        elif split == DatasetSplit.VALIDATION:
            return RNG.choice(self.validation_data, size=size)
        else:
            raise ValueError(
                f"Split has to be a DataSplit enum value!, not {type(split)}")

    def _split_data(self, test_split, validation_split):
        """Splits data into train, test and validation parts.

        Parameters
        ----------
        test_split : float [0,1]
            How many cells should be in the test split.
        validation_split : float [0,1]
            How many cells should be in the validation split.

        Returns
        -------
        list of 3 np.arrays
            train_data, test_data, validation_data
        """
        validation_index = int(self.amount * (1 - validation_split))
        test_index = int(self.amount * (1 - test_split - validation_split))
        return (np.split(self.data, (test_index, validation_index)))

    def _generate_data(self):
        """Method used for example data generation."""
        self.data = np.zeros((self.amount, self.dimension), dtype="float32")
        self._generate_inner_circle()
        self._generate_outer_circle()
        self._shuffle_data()
        self.data[:, 0] += self.offsetX
        self.data[:, 1] += self.offsetY

    def _generate_inner_circle(self):
        """Method used for example data generation."""
        #Generate inner circle (Call only in 2D constructor)
        inner_index_end = int(self.amount * self.center_density)

        self.data[:inner_index_end] = self._generate_circle(
            self.center_radius,
            self.center_scatter,
            inner_index_end
        )
        self.data[:inner_index_end] += 0.1

    def _generate_outer_circle(self):
        """Method used for example data generation."""
        #Generate outer circle (Call only in 2D constructor)
        start_index = int(self.amount * self.center_density) + 1
        length = self.amount - start_index
        self.data[start_index:] = self._generate_circle(self.radius,
                                                        self.outer_scatter,
                                                        length)

    def _shuffle_data(self):
        """Shuffle the data saved in this dataset."""
        RNG.shuffle(self.data)

    def _generate_circle(self, radius, scatter, amount):
        """Return a ring with given values and `amount` points. """
        if self.dimension != 2:
            raise DimensionError(
                "Not able to generate circle in not 2 dimensions")
        ret = np.ndarray((amount, self.dimension))
        theta = np.linspace(0, 2*np.pi, amount)
        ret[:, 0] = radius * np.cos(theta)
        ret[:, 1] = radius * np.sin(theta)
        ret += (scatter
                * (RNG.normal(loc=0.0,
                              scale=scatter,
                              size=(amount, 2)
                              )
                   )
                )
        return ret


class DataDataset(Dataset):
    """Dataset for given data.

    Arguments
    ---------
    data : np.array
        given data
    dimension : int or None
        dimension of data (or None for automatic detection)
    test_split : float
        fraction of data saved for testing
    validation_split : float
        fraction of data saved for validation
    shuffle : bool
        Whether the data should be shuffled before use.
    """

    def __init__(self,
                 data,
                 dimension=None,
                 test_split=0.1,
                 validation_split=0.1,
                 shuffle=True,
                 ):

        self.data = data
        self.amount = data.shape[0]
        if dimension is None:
            self.dimension = data.shape[1]
        else:
            self.dimension = dimension
        if shuffle:
            self._shuffle_data()
        self.train_data, self.test_data, self.validation_data = self._split_data(
            test_split, validation_split)
