import pickle as pkl


if __name__ == '__main__':
    test_file = open('Online Data/test_data_online.pkl', 'rb')
    model_file = open('Online Data/trained_model_online.pkl', 'rb')

    test_data = pkl.load(test_file)
    rec = pkl.load(model_file)

    RMSE = rec.test(test_data[:, 1:])

    print("RMSE for test data = " + str(RMSE))
