import sys
import numpy as np
from CellCNN.Dataset import Dataset
from MNISTGenerator import MNISTGenerator
from CellCNN.InputData import InputData
from random import randint

class MNISTDataset(Dataset):
    def __init__(self,
                    data,
                    dimension = 2,                
                    test_split = 0.1,
                    validation_split = 0.1,
                    offsetX = 0,
                    offsetY = 0
                ):
        """
        Generate an elipse dataset for learning with ratio of
        `center_density` points in the middle from total `amount` points

        Arguments
        ---------
        data : np.array
            given data
        dimension : int
            dimension of data
        test_split : float
            fraction of data saved for testing
        validation_split : float
            fraction of data saved for validation
        """
        self.data = data
        self.amount = data.shape[0]
        self.dimension = dimension
        self.offsetX = offsetX
        self.offsetY = offsetY
        self._shuffle_data()
        self.train_data, self.test_data, self.validation_data = self._split_data(
                                                    test_split, validation_split)

class MNISTInputData(InputData):    
    def __init__(self,
                 odds,
                 chances,
                 multi_cell_size=1500):
        """Prepare mnist data for cellCNN analysis

        Parameters
        ----------
        odds : list of lists
            List of numbers each dataset should ignore (e.g.: [[5,1], [0,], [-1]]).
            Use -1 for ignoring nothing.
            Must have same first dimension as `chances`
        chances : list
            List of probabilities that numbers given in `odds` will be ignored
            e.g.: (50, 20, 0)
            Use 0 for ignoring nothing.
            Must have same first dimension as `odds`
        multi_cell_size : int
            How big should the multicell input be.
            By default 1500
        """
        datas = []
        labels = []
        x,y = MNISTGenerator.load_2D_mnist()
        for j in range(len(odds)):
            this_data = np.zeros((x.shape[0], 2))
            this_labels = np.zeros((x.shape[0]))
            offset = 0
            for i in range(x.shape[0]):
                if y[i] in odds[j] and randint(0,100) <= chances[j]:
                    offset += 1
                else:
                    this_data[i-offset] = x[i]
                    this_labels[i-offset] = y[i]
            if offset > 0:
                this_data = this_data[:-offset]
                this_labels = this_labels[:-offset]
            datas.append(this_data)
            labels.append(this_labels)
        datasets = [MNISTDataset(x) for x in datas]
        return super().__init__(datasets, multi_cell_size)                                                     

if __name__ == "__main__":
    MNISTInputData([[0,], [-1,]],
                    (20, 0))