import pickle


def load_pickle(filename):
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj
