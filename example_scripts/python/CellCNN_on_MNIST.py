# This ensures that CellCNN directory is where the imports are taken from
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.insert(0, dirname(realpath(__name__)) + sep + pardir + sep + "CellCNN")

import tensorflow as tf
from MNISTClasses import MNISTInputData
from Models import CellCNN, SCellCNN
from Dataset import DatasetSplit
def main():    
    MULTI_CELL_SIZE = 1500
    ODDS = [(3, ), (-1,),] # If you don't want to exclude any numbers, just set ODDS[i] to (-1,)
    CHANCES = [20, 0] # how many procent of images in ODD classes should be skipped
    # -> CHANCE = 100 -> very easy, CHANCE = 0 -> impossible
    inp = MNISTInputData(ODDS, CHANCES, multi_cell_size = MULTI_CELL_SIZE, load_from_hd=True)
    FN = 2000
    data = inp.datasets[0].data[:FN]
    X_train, Y_train = inp.get_multi_cell_inputs(10000, DatasetSplit.TRAIN)
    # X_test, Y_test = inp.get_multi_cell_inputs(1000, DatasetSplit.TEST)
    X_eval, Y_eval = inp.get_multi_cell_inputs(1000, DatasetSplit.VALIDATION)

    model = CellCNN(input_shape=(None, MULTI_CELL_SIZE, 2),
                    conv=[64, 64, 7],
                    classes=[inp.length,],
                    lr=0.01)

    model.fit(X_train, Y_train, epochs=1, validation_data=(X_eval, Y_eval),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),])
    model.save("model1.json", "model1.h5")
    second_model = CellCNN.load("model1.json", "model1.h5")
    single_model = SCellCNN(second_model)
    single_model.show_importance(data, scale=True)
if __name__ == "__main__":
    main()