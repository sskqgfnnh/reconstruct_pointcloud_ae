__author__ = 'sskqgfnnh'

import torch
import random
import numpy as np

def test_loss_fun(x, y):
    z = x-y
    z = z * z
    z = np.sum(z, axis=1)
    z = np.sqrt(z)
    return z.mean()

def ColorSample(num):
    if num > 256 ** 3:
        print('color size is more than 256^3')
        return
    color_sample = np.array(random.sample(range(256 ** 3), num))
    b = color_sample % 256
    color_sample = color_sample // 256
    g = color_sample % 256
    color_sample = color_sample // 256
    r = color_sample % 256
    return np.hstack((r.reshape(len(r), -1), g.reshape(len(g), -1), b.reshape(len(b), -1)))

def PlotData(source_data, neural_data, plot_type=0):
    print("PlotCube: source_data shape = {}".format(source_data.shape))
    print("PlotCube: neural_data shape = {}".format(neural_data.shape))
    numX = source_data.shape[0]
    num_col = neural_data.shape[1] // 2
    d = {}
    for i in range(numX):
        if plot_type == 0:
            t = tuple(neural_data[i, 0: num_col])
        if plot_type == 1:
            t = tuple(neural_data[i, num_col + 1:])
        if plot_type == 2:
            t = tuple(neural_data[i, :])
        if t in d:
            d[t] = d[t] + 1
        else:
            d.setdefault(t, 0)
    dict_len = len(d)
    print("PlotCube: dict_len = {}".format(dict_len))
    count = 0
    color_sample = ColorSample(dict_len)
    for k, v in d.items():
        d[k] = np.hstack((count, color_sample[count, :]))
        count = count + 1
    cdata = np.zeros(shape=[numX, 4], dtype=np.int)
    for i in range(numX):
        if plot_type == 0:
            t = tuple(neural_data[i, 0: num_col])
        if plot_type == 1:
            t = tuple(neural_data[i, num_col + 1:])
        if plot_type == 2:
            t = tuple(neural_data[i, :])
        cdata[i, :] = d[t]
    print('color_data shape = {}'.format(cdata.shape))
    fwriter = open("colored_data.txt", "w")
    for i in range(numX):
        my_str = ""
        for j in range(source_data.shape[1]):
            my_str = my_str + str(source_data[i, j]) + " "
        for k in range(cdata.shape[1]):
            my_str = my_str + str(cdata[i, k]) + " "
        my_str = my_str + "\n"
        fwriter.write(my_str)
    fwriter.close()
    return source_data


