import numpy as np
import pandas as pd


def construct_covariance_matrix(data):
    means = np.mean(data, axis=0)
    n = data.shape[0]
    temp = np.matrix(data[0, :]) - np.matrix(means)
    temp_t = temp.transpose()
    covariance_matrix = temp_t*temp
    for i in range(1, n):
        temp = np.matrix(data[i, :]) - np.matrix(means)
        temp_t = temp.transpose()
        covariance_matrix = covariance_matrix + (temp_t*temp)
    covariance_matrix /= n
    return covariance_matrix


def main():
    data = np.genfromtxt('data.txt', delimiter='\t')
    covariance_matrix = construct_covariance_matrix(data)




if __name__ == '__main__':
    main()
