import pandas as pd
import math
import numpy as np
import random

class Node:
    def __init__(self, attribute_no=None, attribute_values=None, child_nodes=None, decision=None):
        self.attribute_no = attribute_no
        self.attribute_values = attribute_values
        self.child_nodes = child_nodes
        self.decision = decision


def count_value(series, val):
    cnt = 0
    for s in series:
        if s == val:
            cnt += 1
    return cnt


class DecisionTree:

    def __init__(self, depth):
        self.depth = depth
        self.root = Node()

    @staticmethod
    def importance(labels, attribute_indices, attribute_values):
        df_temp = pd.concat([attribute_values, labels], axis=1)

        pos_count = count_value(df_temp.iloc[:, df_temp.shape[1] - 1], 1)
        neg_count = count_value(df_temp.iloc[:, df_temp.shape[1] - 1], 0)

        entropy_total = 0
        if pos_count != 0:
            entropy_total = -(float(pos_count) / (pos_count + neg_count)) * math.log(
                float(pos_count) / (pos_count + neg_count), 2)
        if neg_count != 0:
            entropy_total += -(float(neg_count) / (pos_count + neg_count)) * math.log(
                float(neg_count) / (pos_count + neg_count), 2)
        info_gain = []

        for i in attribute_indices:
            temp_attr = attribute_values.iloc[:, i]
            attr_unique_vals = temp_attr.unique()
            gini = entropy_total
            for a in attr_unique_vals:
                temp = df_temp.loc[df_temp.iloc[:, i] == a]

                pos_a_count = count_value(temp.iloc[:, temp.shape[1] - 1], 1)
                neg_a_count = count_value(temp.iloc[:, temp.shape[1] - 1], 0)

                entropy_a = 0
                if pos_a_count != 0:
                    entropy_a = -(float(pos_a_count) / (pos_a_count + neg_a_count)) * math.log(
                        float(pos_a_count) / (pos_a_count + neg_a_count), 2)
                if neg_a_count != 0:
                    entropy_a += -(float(neg_a_count) / (pos_a_count + neg_a_count)) * math.log(
                        float(neg_a_count) / (pos_a_count + neg_a_count), 2)
                gini -= (float(pos_a_count + neg_a_count) / df_temp.shape[0]) * entropy_a
            info_gain.append(gini)
        max_gini_index = info_gain.index(max(info_gain))
        return attribute_indices[max_gini_index]

    def decision_tree_learning_depth(self, labels, attribute_indices, attribute_values, parent_labels, current_depth):
        df = pd.concat([attribute_values, labels], axis=1)
        if labels.size == 0:
            return Node(decision=parent_labels.value_counts().idxmax())
        elif len(labels.unique()) == 1:
            return Node(decision=labels.iloc[0])
        elif len(attribute_indices) == 0:
            return Node(decision=labels.value_counts().idxmax())
        elif current_depth == self.depth:
            return Node(decision=labels.value_counts().idxmax())
        else:
            a = self.importance(labels, attribute_indices, attribute_values)
            attribute_indices.remove(a)
            root = Node(a, attribute_values.iloc[:, a].unique(), [])
            for vk in attribute_values.iloc[:, a].unique():
                df_vk = df.loc[df.iloc[:, a] == vk]
                attribute_indices_copy = attribute_indices[:]
                child_node = self.decision_tree_learning_depth(df_vk.iloc[:, df_vk.shape[1] - 1],
                                                               attribute_indices_copy, df_vk.iloc[:, :df_vk.shape[1] - 1],
                                                               labels,
                                                               current_depth + 1)
                root.child_nodes.append(child_node)
            return root

    def train(self, labels, attribute_values, parent_labels=None):
        attribute_indices = range(0, attribute_values.shape[1])
        self.root = self.decision_tree_learning_depth(labels, attribute_indices, attribute_values, parent_labels, 0)

    def decide(self, attributes):
        current_node = self.root
        while True:
            if current_node.decision is None:
                current_attribute = attributes[current_node.attribute_no]
                branch = np.where(current_node.attribute_values == current_attribute)
                if len(branch[0]) == 0:
                    return random.randint(0, 1)
                current_node = current_node.child_nodes[branch[0][0]]
            else:
                return current_node.decision
