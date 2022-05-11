import numpy as np


PATH = "../data/input.data"
def data_loader():
    x = list()
    y = list()
    with open(PATH, "r") as f:
        for line in f.readlines():
            point = line.split(" ")
            x.append(float(point[0]))
            y.append(float(point[1]))
    return np.array(x, dtype=np.double).reshape(-1, 1), np.array(y, dtype=np.double).reshape(-1, 1)




if __name__ == "__main__":
    x,y  = data_loader()
    print()