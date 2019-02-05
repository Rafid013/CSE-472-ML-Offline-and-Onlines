import pickle as pkl


if __name__ == '__main__':
    test_file = open('test_data.pkl', 'rb')
    model_file = open('trained_model.pkl', 'rb')

    test_data = pkl.load(test_file)
    rec = pkl.load(model_file)

    RMSE = rec.test(test_data[:, 1:])

    print("RMSE for test data = " + str(RMSE))
