import pickle as pkl
from recommender import Recommender


if __name__ == '__main__':
    file1 = open('train_data.pkl', 'rb')
    file2 = open('valid_data.pkl', 'rb')

    train_data = pkl.load(file1)
    valid_data = pkl.load(file2)

    rec = Recommender(0.01, 0.01, 5, 0.001)
    rec.train(train_data[:, 1:])

    RMSE = rec.test(valid_data[:, 1:])
    print(RMSE)
