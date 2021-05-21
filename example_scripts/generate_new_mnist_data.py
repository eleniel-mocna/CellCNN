from os import truncate
from MNISTGenerator import MNISTGenerator
def main():    
    gen = MNISTGenerator(load_from_cache=True)
    gen.train_model()
    gen.get_2D_mnist(load=False)
    gen.scatter_2D_mnist()
    gen.show_encoded_plane()
if __name__ == "__main__":
    main()