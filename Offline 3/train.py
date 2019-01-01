import pickle as pkl
from recommender import Recommender
import math
import numpy as np


if __name__ == '__main__':
    file1 = open('train_data.pkl', 'rb')
    file2 = open('valid_data.pkl', 'rb')

    train_data = pkl.load(file1)
    valid_data = pkl.load(file2)

    train_data = train_data.astype(np.float32)
    valid_data = valid_data.astype(np.float32)

    lambda_u_list = [0.01, 0.1, 1.0, 10.0]
    lambda_v_list = [0.01, 0.1, 1.0, 10.0]
    K_list = [5, 10, 20, 40]

    min_RMSE = math.inf
    min_lambda_u = min_lambda_v = min_K = 0

    iteration = 1
    for lambda_u in lambda_u_list:
        for lambda_v in lambda_v_list:
            for K in K_list:
                print("lambda_u = " + str(lambda_u))
                print("lambda_v = " + str(lambda_v))
                print("K = " + str(K))

                rec = Recommender(lambda_u, lambda_v, K, 0.001)
                rec.train(train_data[:10, 1:])

                RMSE = rec.test(valid_data[:10, 1:])
                print(RMSE)
                if RMSE < min_RMSE:
                    min_RMSE = RMSE
                    min_lambda_u = lambda_u
                    min_lambda_v = lambda_v
                    min_K = K

    print("RMSE minimum for " + str((min_lambda_u, min_lambda_v, min_K)))
