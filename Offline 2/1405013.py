import numpy as np
import matplotlib.pyplot as plt
import random
import math


def normalize(arr):
    size = len(arr)
    tmp = 0
    ret_arr = arr[:]
    for i in range(size):
        tmp += arr[i]
    for i in range(size):
        ret_arr[i] /= tmp
    return ret_arr


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
    weights = k*[1.0/k]
    covariance_mats = []
    random.seed(1)
    for i in range(k):
        temp_m = []
        for j in range(d):
            temp_m.append(random.uniform(0, 1))
        means_list.append(np.matrix(temp_m))

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
    temp4 = math.exp(temp3)
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

    p_list = []
    for i in range(n):
        xi = data[i]
        denominator = 0
        for j in range(k):
            denominator += (weights[j]*Nk(xi, means_list[j], covariance_mats[j]))
        temp = []
        for j in range(k):
            numerator = weights[j]*Nk(xi, means_list[j], covariance_mats[j])
            temp.append(numerator/denominator)
        p_list.append(temp)
    return p_list


def M_step(data, p_list, k):
    n = data.shape[0]

    means_list = []
    covariance_mats = []
    weights = []
    for j in range(k):
        xi = np.matrix(data[0])
        numerator1 = p_list[0][j]*xi
        denominator = p_list[0][j]
        for i in range(1, n):
            xi = np.matrix(data[i])
            numerator1 = numerator1 + (p_list[i][j]*xi)
            denominator += p_list[i][j]

        means = numerator1/denominator
        means_list.append(means)

    for j in range(k):
        means = means_list[j]
        xi = np.matrix(data[0])
        numerator2 = p_list[0][j]*((xi - means).transpose())*(xi - means)
        denominator = 0
        for i in range(1, n):
            xi = np.matrix(data[i])
            numerator2 = numerator2 + (p_list[i][j]*((xi - means).transpose())*(xi - means))
            denominator += p_list[i][j]
        covariance_mat = numerator2/denominator
        covariance_mats.append(covariance_mat)

        weights.append(denominator/n)

    weights = normalize(weights)

    return means_list, weights, covariance_mats


def EM(k, data):
    d = data.shape[1]
    means_list, weights, covariance_mats = initialize_params(k, d)

    log_likelihood_ = log_likelihood(data, means_list, weights, covariance_mats)

    iteration = 1
    while True:
        print('Iteration ' + str(iteration))
        p_list = E_step(data, means_list, weights, covariance_mats)
        means_list, weights, covariance_mats = M_step(data, p_list, k)
        log_likelihood_now = log_likelihood(data, means_list, weights, covariance_mats)
        if math.fabs(log_likelihood_now - log_likelihood_) <= 1e-4:
            break
        log_likelihood_ = log_likelihood_now
        print(log_likelihood_)
        iteration += 1
    return means_list, weights, covariance_mats, p_list


def main():
    data = np.genfromtxt('online_data.txt', delimiter='\t')

    covariance_matrix = construct_covariance_matrix(data)

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    idx = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    transform_matrix = np.matrix(eigen_vectors[:, 0:2].transpose())
    d_vector = np.matrix(data).transpose()
    reduced_sample = (transform_matrix*d_vector).transpose()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(reduced_sample[:, 0].flatten().tolist()[0],
                reduced_sample[:, 1].flatten().tolist()[0])
    ax.set_title('Data Plot')
    fig.tight_layout()
    plt.show()

    means_list, weights, covariance_mats, p_list = EM(4, reduced_sample)
    for i in range(4):
        print(means_list[i].tolist()[0])
    print(weights)

    soft_counts = []
    for j in range(4):
        temp = 0
        for i in range(data.shape[0]):
            temp += p_list[i][j]
        soft_counts.append(temp)
    print(soft_counts)

    red_points_idx = []
    blue_points_idx = []
    green_points_idx = []
    orange_points_idx = []
    for i in range(data.shape[0]):
        j = p_list[i].index(max(p_list[i]))
        if j == 0:
            red_points_idx.append(i)
        elif j == 1:
            blue_points_idx.append(i)
        elif j == 2:
            green_points_idx.append(i)
        else:
            orange_points_idx.append(i)

    red_points = reduced_sample[red_points_idx, :]
    blue_points = reduced_sample[blue_points_idx, :]
    green_points = reduced_sample[green_points_idx, :]
    orange_points = reduced_sample[orange_points_idx, :]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plt.scatter(red_points[:, 0].flatten().tolist()[0],
                red_points[:, 1].flatten().tolist()[0])
    plt.scatter(blue_points[:, 0].flatten().tolist()[0],
                blue_points[:, 1].flatten().tolist()[0])
    plt.scatter(green_points[:, 0].flatten().tolist()[0],
                green_points[:, 1].flatten().tolist()[0])
    plt.scatter(orange_points[:, 0].flatten().tolist()[0],
                orange_points[:, 1].flatten().tolist()[0])
    ax1.set_title('Colored Data Plot')
    fig1.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
