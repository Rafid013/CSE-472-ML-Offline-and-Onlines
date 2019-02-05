import pickle as pkl
from recommender import Recommender
import math


if __name__ == '__main__':
    file1 = open('train_data.pkl', 'rb')
    file2 = open('valid_data.pkl', 'rb')

    out_file = open('out.txt', 'w')

    train_data = pkl.load(file1)
    valid_data = pkl.load(file2)

    file1.close()
    file2.close()

    lambda_u_list = [0.01, 0.1, 1.0, 10.0]
    lambda_v_list = [0.01, 0.1, 1.0, 10.0]
    K_list = [5, 10, 20, 40]

    min_RMSE = math.inf
    min_lambda_u = min_lambda_v = 1
    min_K = 5
    threshold = 0.01

    # iteration = 1
    # for lambda_u in lambda_u_list:
    # for lambda_v in lambda_v_list:
    #    for K in K_list:
    #        print("lambda_u = " + str(lambda_v))
    #        print("lambda_v = " + str(lambda_v))
    #        print("K = " + str(K))

     #       rec = Recommender(lambda_v, lambda_v, K, threshold)
     #       rec.train(train_data[:, 1:])

      #      RMSE = rec.test(valid_data[:, 1:])

    #        out_file.write("Lambda = " + str(lambda_v) + ", K = " + str(K) + ", RMSE = " + str(RMSE) + "\n")
     #       out_file.flush()

      #      if RMSE < min_RMSE:
       #         min_RMSE = RMSE
        #        min_lambda_u = lambda_v
         #       min_lambda_v = lambda_v
          #      min_K = K

    # out_file.write("RMSE minimum for " + str((min_lambda_u, min_lambda_v, min_K)) + "\n")

    # out_file.close()

    for i in range(train_data.shape[0]):
        train_data[i, 0] += valid_data[i, 0]
        for j in range(1, train_data.shape[1]):
            if valid_data[i, j] != 99:
                train_data[i, j] = valid_data[i, j]

    rec = Recommender(min_lambda_u, min_lambda_v, min_K, threshold)
    rec.train(train_data[:, 1:])
    model_file = open('trained_model.pkl', 'wb')
    pkl.dump(rec, model_file)
    model_file.close()
