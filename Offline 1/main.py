import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import decision_tree


def count_value(series, val):
    cnt = 0
    for s in series:
        if s == val:
            cnt += 1
    return cnt


def process_missing_attribute_adult(df):
    for j in [1, 3, 5, 6, 7, 8, 9, 13]:
        if df.iloc[:, j].dtype == np.object_:
            temp = df.iloc[:, j].value_counts().idxmax()
            df.iloc[:, j] = df.iloc[:, j].replace(np.nan, temp, regex=True)
    return df


def process_string_to_int_adult(df):
    for j in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        si = preprocessing.LabelEncoder()
        si.fit(df.iloc[:, j])
        df.iloc[:, j] = si.transform(df.iloc[:, j])
    return df


def binarize_adult(df):
    binarizers = []
    binarizer_columns = []
    for i in [0, 4, 10, 11, 12]:
        df = df.sort_values(by=[i], axis=0, kind='mergesort')
        unique_values = df.iloc[:, i].unique()
        mid_points = [unique_values[0] - 5]
        for k in range(0, len(unique_values) - 1):
            mid_points.append(float(unique_values[k] + unique_values[k + 1]) / 2.0)
        mid_points.append(unique_values[len(unique_values) - 1] + 5)

        pos_count = count_value(df.iloc[:, 14], 1)
        neg_count = count_value(df.iloc[:, 14], 0)

        entropy_total = 0
        if pos_count != 0:
            entropy_total = -(float(pos_count) / (pos_count + neg_count)) * math.log(
                float(pos_count) / (pos_count + neg_count), 2)
        if neg_count != 0:
            entropy_total += -(float(neg_count) / (pos_count + neg_count)) * math.log(
                float(neg_count) / (pos_count + neg_count), 2)

        info_gain = []
        for k in range(0, len(mid_points)):
            mid_point = mid_points[k]
            pos_count_before = neg_count_before = pos_count_after = neg_count_after = 0
            for j in range(0, df.shape[0]):
                if df.iloc[j, 0] <= mid_point:
                    if df.iloc[j, 14] == 1:
                        pos_count_before += 1
                    else:
                        neg_count_before += 1
                else:
                    pos_count_after = pos_count - pos_count_before
                    neg_count_after = neg_count - neg_count_before

            entropy_before = entropy_after = 0
            if pos_count_before + neg_count_before > 0:
                temp1 = float(pos_count_before) / (pos_count_before + neg_count_before)
                temp2 = float(neg_count_before) / (pos_count_before + neg_count_before)
                if temp1 != 0:
                    entropy_before = -temp1 * math.log(temp1, 2)
                if temp2 != 0:
                    entropy_before += -temp2 * math.log(temp2, 2)

            if pos_count_after + neg_count_after > 0:
                temp1 = float(pos_count_after) / (pos_count_after + neg_count_after)
                temp2 = float(neg_count_after) / (pos_count_after + neg_count_after)
                if temp1 != 0:
                    entropy_after = -temp1 * math.log(temp1, 2)
                if temp2 != 0:
                    entropy_after += -temp2 * math.log(temp2, 2)

            gini = entropy_total - (float(pos_count_before + neg_count_before) / df.shape[0]) * entropy_before
            gini -= (float(pos_count_after + neg_count_after) / df.shape[0]) * entropy_after
            info_gain.append(gini)
        temp = mid_points[info_gain.index(max(info_gain))]
        binarizer = preprocessing.Binarizer(threshold=temp).fit([df.iloc[:, i]])
        binarizers.append(binarizer)
        binarizer_columns.append(i)
        df.iloc[:, i] = binarizer.transform([df.iloc[:, i]])[0]

    temp = (max(df.iloc[:, 2]) + min(df.iloc[:, 2])) / 2
    binarizer = preprocessing.Binarizer(threshold=temp).fit([df.iloc[:, 2]])
    binarizers.append(binarizer)
    binarizer_columns.append(2)
    df.iloc[:, 2] = binarizer.transform([df.iloc[:, 2]])[0]
    return df, binarizers, binarizer_columns


def binarize_adult_test(binarizers, binarizers_columns, df):
    for i, binarizer in zip(binarizers_columns, binarizers):
        df.iloc[:, i] = binarizer.transform([df.iloc[:, i]])[0]
    return df


df_adult_train = pd.read_csv('adult_data.csv', delimiter=',', header=None, na_values=' ?')

df_adult_train = process_missing_attribute_adult(df_adult_train)
df_adult_train = process_string_to_int_adult(df_adult_train)
df_adult_train, binarizers_adult, binarizers_adult_columns = binarize_adult(df_adult_train)

df_adult_test = pd.read_csv('adult_test.csv', delimiter=',', header=None, na_values=' ?')

df_adult_test = process_missing_attribute_adult(df_adult_test)
df_adult_test = process_string_to_int_adult(df_adult_test)
df_adult_test = binarize_adult_test(binarizers_adult, binarizers_adult_columns, df_adult_test)

dt = decision_tree.DecisionTree(14)
dt.decision_tree_learning(df_adult_train.iloc[:, 14], range(0, 14), df_adult_train.iloc[:, 0:14])



