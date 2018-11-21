from sklearn import preprocessing
import numpy as np
import math


def count_value(series, val):
    cnt = 0
    for s in series:
        if s == val:
            cnt += 1
    return cnt


def process_missing_attribute_credit(df):
    for j in range(0, 30):
        temp = df.iloc[:, j].mean()
        df.iloc[:, j] = df.iloc[:, j].replace(np.nan, temp, regex=True)
    return df


def binarize_credit(df):
    binarizers = []
    binarizer_columns = []
    for i in range(0, 0):
        df = df.sort_values(by=[i], axis=0, kind='mergesort')
        unique_values = df.iloc[:, i].unique()
        mid_points = [unique_values[0] - 5]
        for k in range(0, len(unique_values) - 1):
            mid_points.append(float(unique_values[k] + unique_values[k + 1]) / 2.0)
        mid_points.append(unique_values[len(unique_values) - 1] + 5)

        pos_count = count_value(df.iloc[:, 20], 1)
        neg_count = count_value(df.iloc[:, 20], 0)

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
                    if df.iloc[j, 20] == 1:
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

    for i in [0, 30]:
        temp = (max(df.iloc[:, i]) + min(df.iloc[:, i])) / 2
        binarizer = preprocessing.Binarizer(threshold=temp).fit([df.iloc[:, i]])
        binarizers.append(binarizer)
        binarizer_columns.append(i)
        df.iloc[:, i] = binarizer.transform([df.iloc[:, i]])[0]
    return df, binarizers, binarizer_columns
