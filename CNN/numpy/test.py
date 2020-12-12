import numpy as np
from Conv import ReLU

if __name__ == '__main__':
    a = np.empty([3, 2], dtype=int)
    a[0][0] = -1
    print(a)
    a = ReLU(a)
    print(a)
