# This ensures that CellCNN directory is where the imports are taken from
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.insert(0, dirname(realpath(__name__)) + sep + pardir + sep + "CellCNN")

from MNISTGenerator import MNISTGenerator
def main():    
    gen = MNISTGenerator(load_from_cache=True)
    gen.train_model()
    gen.get_2D_mnist(load=False)
    gen.scatter_2D_mnist()
    gen.show_encoded_plane()
if __name__ == "__main__":
    main()