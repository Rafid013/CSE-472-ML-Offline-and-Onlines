import time

import pandas as pd
from sklearn import metrics, model_selection
import preprocess as tp
import decision_tree
import adaboost


start_preprocessing = time.time()
print 'Preprocessing started...'
df_telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', delimiter=',', header=None, na_values='\s+', skiprows=1)
df_telco.iloc[:, 5] = pd.to_numeric(df_telco.iloc[:, 5])
df_telco.iloc[:, 18] = pd.to_numeric(df_telco.iloc[:, 18])
df_telco.iloc[:, 19] = pd.to_numeric(df_telco.iloc[:, 19], errors='coerce')
df_telco = tp.process_missing_label(df_telco)
df_telco = tp.process_missing_attribute(df_telco, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
print 'Missing values handled'
df_telco = tp.process_string_to_int(df_telco, [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
print 'String converted to integer labels'
df_telco, binarizers_telco, binarizers_telco_columns = tp.binarize(df_telco, [5, 18, 19])
print 'Continuous values binarized'
df_telco.reset_index(drop=True)
df_telco.to_csv('Preprocessed_Telco.csv', sep=',')
print 'Preprocessing Finished...'
end_preprocessing = time.time()

print "Preprocessing took " + str(float(end_preprocessing - start_preprocessing)/60) + " min"
print df_telco
# df_telco = pd.read_csv('Preprocessed_Telco.csv', delimiter=',', header=None)
# print df_telco

df_telco_train, df_telco_test = model_selection.train_test_split(df_telco, test_size=0.20)

start_training = time.time()
dt = decision_tree.DecisionTree(20)
dt.train(df_telco_train.iloc[:, 20], df_telco_train.iloc[:, :20])

# dt = adaboost.AdaBoost(df_telco_train, decision_tree.DecisionTree, 20)
# dt.train()
end_training = time.time()

print "Training took " + str(float(end_training - start_training)/60) + " min"

results = []
for smpl in range(0, df_telco_train.shape[0]):
    results.append(dt.decide(df_telco_train.iloc[smpl, :20].tolist()))

li = metrics.confusion_matrix(df_telco_train.iloc[:, 20].tolist(), results).ravel()

tn = li[0]
fp = li[1]
fn = li[2]
tp = li[3]

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


results = []
for smpl in range(0, df_telco_test.shape[0]):
    results.append(dt.decide(df_telco_test.iloc[smpl, :20].tolist()))

tn, fp, fn, tp = metrics.confusion_matrix(df_telco_test.iloc[:, 20].tolist(), results).ravel()
print tn, fp, fn, tp

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
