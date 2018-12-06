import numpy as np
import matplotlib.pyplot as plt
import random
import math


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


def initialize_params(k, d):
    means_list = []
    weights = []
    covariance_mats = []
    for i in range(k):
        temp_m = []
        for j in range(d):
            temp_m.append(random.uniform(0, 1))
        means_list.append(np.matrix(temp_m))

        weights.append(random.uniform(0, 1))

        temp1 = []
        for j1 in range(d):
            temp2 = []
            for j2 in range(d):
                temp2.append(random.uniform(0, 1))
            temp1.append(temp2)
        covariance_mats.append(np.matrix(temp1))
    return means_list, weights, covariance_mats


def Nk(x, means, covariance_mat):
    d = x.shape[1]
    temp1 = math.sqrt(math.pow(2*math.pi, d)*math.fabs(np.linalg.det(covariance_mat)))
    temp2 = np.matrix(x) - np.matrix(means)
    temp3 = -0.5 * temp2 * np.linalg.inv(covariance_mat) * temp2.transpose()
    try:
        temp4 = math.exp(temp3)
    except OverflowError:
        if temp3 > 0:
            temp4 = 1
        else:
            temp4 = 0
    result = temp4/temp1
    return result


def log_likelihood(data, means_list, weights, covariance_mats):
    n = data.shape[0]
    k = len(weights)

    log_likelihood_ = 0
    for i in range(n):
        temp = 0
        for j in range(k):
            temp += (weights[j]*Nk(data[i], means_list[j], covariance_mats[j]))
        log_likelihood_ += math.log(temp, math.e)
    return log_likelihood_


def E_step(data, means_list, weights, covariance_mats):
    n = data.shape[0]
    k = len(weights)

    p_list = n*[k*[0]]

    for i in range(n):
        xi = data[i]
        denominator = 0
        for j in range(k):
            denominator += (weights[j]*Nk(xi, means_list[j], covariance_mats[j]))

        for j in range(k):
            numerator = weights[j]*Nk(xi, means_list[j], covariance_mats[j])
            p_list[i][j] = numerator/denominator

    return p_list


def M_step(data, p_list, k):
    n = data.shape[0]
    d = data.shape[1]

    means_list = []
    covariance_mats = []
    weights = []
    for j in range(k):
        numerator1 = np.matrix(d*[0])
        denominator = 0
        for i in range(n):
            xi = np.matrix(data[i])
            numerator1 = numerator1 + (p_list[i][j]*xi)
            denominator += p_list[i][j]

        means = numerator1/denominator
        means_list.append(means)

    for j in range(k):
        means = means_list[j]
        numerator2 = np.matrix(d * [d * [0]])
        denominator = 0
        for i in range(n):
            xi = np.matrix(data[i])
            numerator2 = numerator2 + (p_list[i][j]*((xi - means).transpose())*(xi - means))
            denominator += p_list[i][j]
        covariance_mat = numerator2/denominator
        covariance_mats.append(covariance_mat)

        weights.append(denominator/n)

    return means_list, weights, covariance_mats


def EM(k, data):
    d = data.shape[1]
    means_list, weights, covariance_mats = initialize_params(k, d)

    log_likelihood_ = log_likelihood(data, means_list, weights, covariance_mats)

    while True:
        p_list = E_step(data, means_list, weights, covariance_mats)
        means_list, weights, covariance_mats = M_step(data, p_list, k)
        log_likelihood_now = log_likelihood(data, means_list, weights, covariance_mats)
        if math.fabs(log_likelihood_now - log_likelihood_) <= 0.01:
            break
        log_likelihood_ = log_likelihood_now
    return means_list, weights, covariance_mats


def main():
    data = np.genfromtxt('data.txt', delimiter='\t')

    covariance_matrix = construct_covariance_matrix(data)

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    idx = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    transform_matrix = np.matrix(eigen_vectors[:, 0:2].transpose())
    d_vector = np.matrix(data).transpose()
    reduced_sample = (transform_matrix*d_vector).transpose()

    fig, ax = plt.subplots(figsize=(5, 3))
    plt.scatter(reduced_sample[:, 0].flatten().tolist()[0],
                reduced_sample[:, 1].flatten().tolist()[0])
    ax.set_title('Data Plot')
    fig.tight_layout()
    plt.show()

    means_list, weights, covariance_mats = EM(3, reduced_sample)
    print(means_list)
    print(weights)
    print(covariance_mats)


if __name__ == '__main__':
    main()
