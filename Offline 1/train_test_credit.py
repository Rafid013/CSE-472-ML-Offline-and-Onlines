import time

import pandas as pd
from sklearn import metrics, model_selection
import preprocess as cp
import decision_tree
import adaboost

prepro = 0
if prepro == 1:
    start_preprocessing = time.time()
    print('Preprocessing started...')
    df_credit_temp = pd.read_csv('creditcard.csv', delimiter=',', header=None, na_values='\s+', skiprows=1)
    df_credit_pos = df_credit_temp.loc[df_credit_temp.iloc[:, df_credit_temp.shape[1] - 1] == 1]
    df_credit_neg = df_credit_temp.loc[df_credit_temp.iloc[:, df_credit_temp.shape[1] - 1] == 0]
    df_credit_neg = df_credit_neg.sample(n=20000, replace=False)
    df_credit = pd.concat([df_credit_neg, df_credit_pos], axis=0)
    df_credit = cp.process_missing_label(df_credit)
    df_credit = cp.process_missing_attribute(df_credit, range(0, df_credit.shape[1] - 1))
    print('Missing values handled')
    df_credit, binarizers_credit, binarizers_credit_columns = cp.binarize(df_credit, range(0, df_credit.shape[1] - 1))
    print('Continuous values binarized')
    df_credit = df_credit.reset_index(drop=True)
    df_credit.to_csv('Preprocessed_Credit.csv', sep=',')
    print('Preprocessing Finished...')
    end_preprocessing = time.time()

    print("Preprocessing took " + str(float(end_preprocessing - start_preprocessing) / 60) + " min")
    print()
    print()
else:
    df_credit = pd.read_csv('Preprocessed_Credit.csv', delimiter=',', header=None)

df_credit_train, df_credit_test = model_selection.train_test_split(df_credit, test_size=0.20)

start_training = time.time()
# dt = decision_tree.DecisionTree(df_credit.shape[1] - 1)
# dt.train(df_credit_train.iloc[:, df_credit.shape[1] - 1], df_credit_train.iloc[:, :df_credit.shape[1] - 1])

dt = adaboost.AdaBoost(df_credit_train, decision_tree.DecisionTree, 20)
dt.train()
end_training = time.time()

print("Training took " + str(float(end_training - start_training) / 60) + " min")
print()
print()
results = []
for smpl in range(0, df_credit_train.shape[0]):
    results.append(dt.decide(df_credit_train.iloc[smpl, :df_credit.shape[1] - 1].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_credit_train.iloc[:, df_credit.shape[1] - 1].tolist(), results).ravel()

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
for smpl in range(0, df_credit_test.shape[0]):
    results.append(dt.decide(df_credit_test.iloc[smpl, :df_credit.shape[1] - 1].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_credit_test.iloc[:, df_credit.shape[1] - 1].tolist(), results).ravel()

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
