import numpy as np

PoolingOption = ("MAX", "AVERAGE")


class Conv:
    def __init__(self, kernel, stride=1, padding="SAME"):
        self.stride = stride
        self.kernel = kernel

    def forward(self, input):
        h, w = input.shape
        filter_h, filter_w = self.kernel.shape
        output_h = (h - filter_h) / self.stride + 1
        output_w = (w - filter_w) / self.stride + 1
        output = np.zeros((output_h, output_w))

        for i in range(0, output_h, self.stride):
            for j in range(0, output_w, self.stride):
                for m in range(filter_h):
                    for n in range(filter_w):
                        output[i][j] += input[i+m][j+n] * self.kernel.filter[i][j]

        return output


class Filter:
    def __init__(self, shape=(3, 3)):
        self.shape = shape
        self.filter = np.random.random(shape)


class Pooling:
    def __init__(self, shape=(2, 2), stride=2, mode="MAX"):
        self.shape = shape
        if mode in PoolingOption:
            self.mode = mode
        else:
            print("Unknown Mode for Pooling. Use default MaxPooling.", str(PoolingOption))
            self.mode = PoolingOption[0]
        self.stride = stride

    def forward(self, input):
        if self.mode == "MAX":
            return self.max_pool(input)
        elif self.mode == "AVERAGE":
            return self.average_pool(input)

    def max_pool(self, input):
        h, w = input.shape
        filter_h, filter_w = self.shape
        output_h = (h - filter_h) / self.stride + 1
        output_w = (w - filter_w) / self.stride + 1
        output = np.zeros((output_h, output_w))

        for i in range(0, output_h, self.stride):
            for j in range(0, output_w, self.stride):
                for m in range(filter_h):
                    for n in range(filter_w):
                        output[i][j] = max(input[i+m][j+n])

        return output

    def average_pool(self, input):
        h, w = input.shape
        filter_h, filter_w = self.shape
        output_h = (h - filter_h) / self.stride + 1
        output_w = (w - filter_w) / self.stride + 1
        output = np.zeros((output_h, output_w))

        for i in range(0, output_h, self.stride):
            for j in range(0, output_w, self.stride):
                for m in range(filter_h):
                    for n in range(filter_w):
                        output[i][j] += input[i + m][j + n]
                output[i][j] /= filter_w + filter_h

        return output


def ReLU(input):
    h, w = input.shape
    output = np.zeros(input.shape)

    for i in range(h):
        for j in range(w):
            output[i][j] = max(0, input[i][j])

    return output
