import numpy as np
import random as rand
import pickle as pkl


def split(data):
    train = np.copy(data)
    valid = np.copy(data)
    test = np.copy(data)

    total_row = train.shape[0]
    total_col = train.shape[1]

    for i in range(total_row):
        print("Splitting row " + str(i))

        rating_cnt = train[i, 0]

        train_rating_cnt = int(rating_cnt*0.6)
        valid_rating_cnt = int(rating_cnt*0.2)
        test_rating_cnt = rating_cnt - train_rating_cnt - valid_rating_cnt

        train[i, 0] = train_rating_cnt
        valid[i, 0] = valid_rating_cnt
        test[i, 0] = test_rating_cnt

        for j in range(1, total_col):
            if train[i, j] != 99:
                while True:
                    temp = rand.randint(1, 3)
                    if temp == 1 and train_rating_cnt == 0:
                        continue
                    elif temp == 2 and valid_rating_cnt == 0:
                        continue
                    elif temp == 3 and test_rating_cnt == 0:
                        continue
                    else:
                        if temp == 1:
                            train_rating_cnt -= 1
                            valid[i, j] = test[i, j] = 99
                        elif temp == 2:
                            valid_rating_cnt -= 1
                            valid[i, j] = train[i, j]
                            train[i, j] = test[i, j] = 99
                        else:
                            test_rating_cnt -= 1
                            test[i, j] = train[i, j]
                            train[i, j] = valid[i, j] = 99
                        break
    return train, valid, test


if __name__ == '__main__':
    data_arr = np.genfromtxt(fname='data.csv', dtype=np.float32, delimiter=',')
    train_data, valid_data, test_data = split(data_arr)
    print(train_data)
    print(valid_data)
    print(test_data)

    file1 = open('train_data.pkl', 'wb')
    file2 = open('valid_data.pkl', 'wb')
    file3 = open('test_data.pkl', 'wb')

    pkl.dump(train_data, file1)
    pkl.dump(valid_data, file2)
    pkl.dump(test_data, file3)

    file1.close()
    file2.close()
    file3.close()
