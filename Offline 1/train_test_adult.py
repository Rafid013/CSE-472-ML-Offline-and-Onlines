import time

import pandas as pd
from sklearn import metrics

import adult_process as ap
import decision_tree


start_preprocessing = time.time()
df_adult_train = pd.read_csv('adult_data.csv', delimiter=',', header=None, na_values=' ?')
df_adult_train = ap.process_missing_attribute_adult(df_adult_train)
df_adult_train = ap.process_string_to_int_adult(df_adult_train)
df_adult_train, binarizers_adult, binarizers_adult_columns = ap.binarize_adult(df_adult_train)
end_preprocessing = time.time()

print "Preprocessing on training data took " + str(float(end_preprocessing - start_preprocessing)/60) + " min"

start_training = time.time()
dt = decision_tree.DecisionTree(14)
dt.decision_tree_learning(df_adult_train.iloc[:, 14], range(0, 14), df_adult_train.iloc[:, :14])
end_training = time.time()

print "Training took " + str(float(end_training - start_training)/60) + " min"

results = []
for smpl in range(0, df_adult_train.shape[0]):
    results.append(dt.decide(df_adult_train.iloc[smpl, :14].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_adult_train.iloc[:, 14].tolist(), results).ravel()

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


print
print
print "Result on training data:"
print "#############"
print 'Accuracy = ' + str(acc)
print 'True Positive Rate = ' + str(tpr)
print 'True Negative Rate = ' + str(tnr)
print 'Precision = ' + str(prc)
print 'False Discovery Rate = ' + str(fdr)
print 'F1 Score = ' + str(f1s)
print "#############"
print
print
print


start_preprocessing = time.time()
df_adult_test = pd.read_csv('adult_test.csv', delimiter=',', header=None, na_values=' ?')
df_adult_test = ap.process_missing_attribute_adult(df_adult_test)
df_adult_test = ap.process_string_to_int_adult(df_adult_test)
df_adult_test = ap.binarize_adult_test(binarizers_adult, binarizers_adult_columns, df_adult_test)
end_preprocessing = time.time()

print "Preprocessing on test data took " + str(float(end_preprocessing - start_preprocessing)/60) + " min"


results = []
for smpl in range(0, df_adult_test.shape[0]):
    results.append(dt.decide(df_adult_test.iloc[smpl, :14].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_adult_test.iloc[:, 14].tolist(), results).ravel()

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


print "Result on test data:"
print "#############"
print 'Accuracy = ' + str(acc)
print 'True Positive Rate = ' + str(tpr)
print 'True Negative Rate = ' + str(tnr)
print 'Precision = ' + str(prc)
print 'False Discovery Rate = ' + str(fdr)
print 'F1 Score = ' + str(f1s)
print "#############"
print
print
print