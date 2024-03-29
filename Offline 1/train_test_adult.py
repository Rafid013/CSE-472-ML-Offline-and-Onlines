import time

import pandas as pd
from sklearn import metrics

import preprocess as ap
import decision_tree
import adaboost


prepro = 0
if prepro == 1:
    start_preprocessing = time.time()
    print('Preprocessing started...')
    df_adult_train = pd.read_csv('adult_data.csv', delimiter=',', header=None, na_values=' ?')
    df_adult_train = ap.process_missing_label(df_adult_train)
    df_adult_train = ap.process_missing_attribute(df_adult_train, [1, 3, 5, 6, 7, 8, 9, 13])
    print('Missing values handled')
    df_adult_train = ap.process_string_to_int(df_adult_train, [1, 3, 5, 6, 7, 8, 9, 13, 14])
    print('String converted to integer labels')
    df_adult_train, binarizers_adult, binarizers_adult_columns = ap.binarize(df_adult_train, [0, 2, 4, 10, 11, 12])
    print('Continuous values binarized')
    df_adult_train = df_adult_train.reset_index(drop=True)
    df_adult_train.to_csv('Preprocessed_Adult_Train.csv', sep=',')
    end_preprocessing = time.time()

    print("Preprocessing training data took " + str(float(end_preprocessing - start_preprocessing) / 60) + " min")

    start_preprocessing = time.time()
    df_adult_test = pd.read_csv('adult_test.csv', delimiter=',', header=None, na_values=' ?')
    df_adult_test = ap.process_missing_label(df_adult_test)
    df_adult_test = ap.process_missing_attribute(df_adult_test, [1, 3, 5, 6, 7, 8, 9, 13])
    df_adult_test = ap.process_string_to_int(df_adult_test, [1, 3, 5, 6, 7, 8, 9, 13, 14])
    df_adult_test = ap.binarize_test(binarizers_adult, binarizers_adult_columns, df_adult_test)
    df_adult_test = df_adult_test.reset_index(drop=True)
    df_adult_test.to_csv('Preprocessed_Adult_Test.csv', sep=',')
    end_preprocessing = time.time()

    print("Preprocessing on test data took " + str(float(end_preprocessing - start_preprocessing) / 60) + " min")
else:
    df_adult_train = pd.read_csv('Preprocessed_Adult_Train.csv', delimiter=',', header=None)
    df_adult_test = pd.read_csv('Preprocessed_Adult_Test.csv', delimiter=',', header=None)


start_training = time.time()
# dt = decision_tree.DecisionTree(df_adult_train.shape[1] - 1)
# dt.train(df_adult_train.iloc[:, df_adult_train.shape[1] - 1], df_adult_train.iloc[:, :df_adult_train.shape[1] - 1])

dt = adaboost.AdaBoost(df_adult_train, decision_tree.DecisionTree, 20)
dt.train()
end_training = time.time()

print("Training took " + str(float(end_training - start_training) / 60) + " min")
print()
print()
results = []
for smpl in range(0, df_adult_train.shape[0]):
    results.append(dt.decide(df_adult_train.iloc[smpl, :df_adult_train.shape[1] - 1].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_adult_train.iloc[:, df_adult_train.shape[1] - 1].tolist(), results).ravel()

print('True positive = ' + str(tp))
print('True negative = ' + str(tn))
print('False positive = ' + str(fp))
print('False positive = ' + str(fn))

total = tn + fp + fn + tp

acc = float(tp + tn)/total

if tp + fn == 0:
    tpr = 'undefined'
else:
    tpr = float(tp)/(tp + fn)

if tn + fp == 0:
    tnr = 'undefined'
else:
    tnr = float(tn) / (tn + fp)

if tp + fp == 0:
    prc = 'undefined'
    fdr = 'undefined'
else:
    prc = float(tp)/(tp + fp)
    fdr = float(fp) / (tp + fp)

if tpr != 'undefined' and prc != 'undefined':
    f1s = 2 / (1/tpr + 1/prc)
else:
    f1s = 'undefined'

print()
print()
print("Result on training data:")
print("#############")
print('Accuracy = ' + str(acc))
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('Precision = ' + str(prc))
print('False Discovery Rate = ' + str(fdr))
print('F1 Score = ' + str(f1s))
print("#############")
print()
print()
print()
results = []
for smpl in range(0, df_adult_test.shape[0]):
    results.append(dt.decide(df_adult_test.iloc[smpl, :df_adult_test.shape[1] - 1].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_adult_test.iloc[:, df_adult_test.shape[1] - 1].tolist(), results).ravel()

print('True positive = ' + str(tp))
print('True negative = ' + str(tn))
print('False positive = ' + str(fp))
print('False positive = ' + str(fn))

total = tn + fp + fn + tp

acc = float(tp + tn)/total

if tp + fn == 0:
    tpr = 'undefined'
else:
    tpr = float(tp)/(tp + fn)

if tn + fp == 0:
    tnr = 'undefined'
else:
    tnr = float(tn) / (tn + fp)

if tp + fp == 0:
    prc = 'undefined'
    fdr = 'undefined'
else:
    prc = float(tp)/(tp + fp)
    fdr = float(fp) / (tp + fp)

if tpr != 'undefined' and prc != 'undefined':
    f1s = 2 / (1/tpr + 1/prc)
else:
    f1s = 'undefined'

print()
print()
print("Result on test data:")
print("#############")
print('Accuracy = ' + str(acc))
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('Precision = ' + str(prc))
print('False Discovery Rate = ' + str(fdr))
print('F1 Score = ' + str(f1s))
print("#############")
print()
print()
print()
