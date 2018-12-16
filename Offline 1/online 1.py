import time

import pandas as pd
from sklearn import metrics, model_selection

import preprocess as ap
import decision_tree
import adaboost


def measure_performance(true_pos, true_neg, false_pos, false_neg, weights):
    tp1 = true_pos
    tn1 = true_neg
    fp1 = false_pos
    fn1 = false_neg

    a = weights[0]
    b = weights[1]
    c = weights[2]
    d = weights[3]

    print('True positive = ' + str(tp1))
    print('True negative = ' + str(tn1))
    print('False positive = ' + str(fp1))
    print('False positive = ' + str(fn1))

    total = b*tn1 + c*fp1 + d*fn1 + a*tp1

    acc = float(a*tp + b*tn) / total
    tpr = float(a*tp) / (a*tp + d*fn)
    tnr = float(b*tn) / (b*tn + c*fp)
    prc = float(a*tp) / (a*tp + c*fp)
    fdr = float(c*fp) / (a*tp + c*fp)
    f1s = 2 / (1 / tpr + 1 / prc)

    print()
    print()
    print()
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


start_preprocessing = time.time()
print('Preprocessing started...')
df = pd.read_csv('online1_data.csv', delimiter=',', header=None, na_values='\s+', skiprows=1)
df = ap.process_missing_label(df)
df = ap.process_missing_attribute(df, range(0, 8))
print('Missing values handled')
df = df.reset_index(drop=True)
end_preprocessing = time.time()

print("Preprocessing training data took " + str(float(end_preprocessing - start_preprocessing) / 60) + " min")

df_train, df_test = model_selection.train_test_split(df, test_size=0.20)

start_training = time.time()
# dt = decision_tree.DecisionTree(df_train.shape[1] - 1)
# dt.train(df_train.iloc[:, df_train.shape[1] - 1], df_train.iloc[:, :df_train.shape[1] - 1])

dt = adaboost.AdaBoost(df_train, decision_tree.DecisionTree, 5)
dt.train()
end_training = time.time()

print("Training took " + str(float(end_training - start_training) / 60) + " min")
print()
print()
results = []
for smpl in range(0, df_train.shape[0]):
    results.append(dt.decide(df_train.iloc[smpl, :df_train.shape[1] - 1].tolist()))


returned = metrics.confusion_matrix(df_train.iloc[:, df_train.shape[1] - 1].tolist(), results)
tp = returned[0][0]
fp = returned[0][1]
tn = returned[1][1]
fn = returned[1][0]

measure_performance(tp, tn, fp, fn, [1, 1, 5, 5])

results = []
for smpl in range(0, df_test.shape[0]):
    results.append(dt.decide(df_test.iloc[smpl, :df_test.shape[1] - 1].tolist()))

returned = metrics.confusion_matrix(df_test.iloc[:, df_test.shape[1] - 1].tolist(), results)
tp = returned[0][0]
fp = returned[0][1]
tn = returned[1][1]
fn = returned[1][0]

measure_performance(tp, tn, fp, fn, [1, 1, 5, 5])
