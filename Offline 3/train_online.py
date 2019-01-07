import pickle as pkl
from recommender import Recommender
import math


if __name__ == '__main__':
    file1 = open('Online Data/train_data_online.pkl', 'rb')
    file2 = open('Online Data/valid_data_online.pkl', 'rb')

    train_data = pkl.load(file1)
    valid_data = pkl.load(file2)

    file1.close()
    file2.close()

    # lambda_u_list = [0.01, 0.1, 1.0, 10.0]
    # lambda_v_list = [0.01, 0.1, 1.0, 10.0]
    # K_list = [5, 10, 20, 40]

    min_RMSE = math.inf
    min_lambda_u = min_lambda_v = 1
    min_K = 5
    threshold = 0.01

    for i in range(train_data.shape[0]):
        train_data[i, 0] += valid_data[i, 0]
        for j in range(1, train_data.shape[1]):
            if valid_data[i, j] != 99:
                train_data[i, j] = valid_data[i, j]

    rec = Recommender(min_lambda_u, min_lambda_v, min_K, threshold)
    rec.train(train_data[:, 1:])
    model_file = open('Online Data/trained_model_online.pkl', 'wb')
    pkl.dump(rec, model_file)
    model_file.close()
