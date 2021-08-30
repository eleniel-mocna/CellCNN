import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

RNG = np.random.default_rng(211)


class DatasetSplit(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class DimensionError(ValueError):
    pass


def plot_datasets(list_of_datasets, list_of_styles=("g.", "r.", "b.")):
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

    def plot_data(self, style='b.', show=True, verbose=1):
        """ Create a plot showing this data. """
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
        """Randomly choose `size` of values from dataset. """
        if split == DatasetSplit.TRAIN:
            return RNG.choice(self.train_data, size=size)
        elif split == DatasetSplit.TEST:
            return RNG.choice(self.test_data, size=size)
        elif split == DatasetSplit.VALIDATION:
            return RNG.choice(self.validation_data, size=size)
        else:
            raise ValueError(f"Split has to be a DataSplit enum value!, not {type(split)}")

    def _split_data(self, test_split, validation_split):
        """
        Splits data into train, test and validation parts and returns them as
        (train_data, test_data, validation_data)
        """
        validation_index = int(self.amount * (1 - validation_split))
        test_index = int(self.amount * (1 - test_split - validation_split))
        return (np.split(self.data, (test_index, validation_index)))

    def _generate_data(self):
        self.data = np.zeros((self.amount, self.dimension), dtype="float32")
        self._generate_inner_circle()
        self._generate_outer_circle()
        self._shuffle_data()
        self.data[:, 0] += self.offsetX
        self.data[:, 1] += self.offsetY

    def _generate_inner_circle(self):
        #Generate inner circle (Call only in 2D constructor)
        inner_index_end = int(self.amount * self.center_density)

        self.data[:inner_index_end] = self._generate_circle(
            self.center_radius,
            self.center_scatter,
            inner_index_end
        )
        self.data[:inner_index_end] += 0.1

    def _generate_outer_circle(self):
        #Generate outer circle (Call only in 2D constructor)
        start_index = int(self.amount * self.center_density) + 1
        length = self.amount - start_index
        self.data[start_index:] = self._generate_circle(self.radius,
                                                        self.outer_scatter,
                                                        length)

    def _shuffle_data(self):
        RNG.shuffle(self.data)

    def _generate_circle(self, radius, scatter, amount):
        """ Return a ring with given values and `amount` points. """
        if self.dimension != 2:
            raise DimensionError("Not able to generate circle in 2 dimensions")
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
    def __init__(self,
                 data,
                 dimension=None,
                 test_split=0.1,
                 validation_split=0.1,
                 offsetX=0,
                 offsetY=0,
                 shuffle=True,
                 ):
        """

        Arguments
        ---------
        data : np.array
            given data
        dimension : int
            dimension of data (or None for automatic detection)
        test_split : float
            fraction of data saved for testing
        validation_split : float
            fraction of data saved for validation
        """
        self.data = data
        self.amount = data.shape[0]
        if dimension is None:
            self.dimension = data.shape[1]
        else:
            self.dimension = dimension
        self.offsetX = offsetX
        self.offsetY = offsetY
        if shuffle:
            self._shuffle_data()
        self.train_data, self.test_data, self.validation_data = self._split_data(
            test_split, validation_split)
