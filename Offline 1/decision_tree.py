import pandas as pd
import math


class Node:
    def __init__(self, attribute_no=None, attribute_values=None, child_nodes=None, decision=None):
        self.attribute_no = attribute_no
        self.attribute_values = attribute_values
        self.child_nodes = child_nodes
        self.decision = decision


class DecisionTree:

    def __init__(self, depth):
        self.depth = depth
        self.root = Node()

    @staticmethod
    def importance(labels, attribute_indices, attribute_values):
        df_temp = pd.concat([attribute_values, labels])

        pos_count = df_temp.iloc[:, df_temp.shape[1] - 1].value_counts()[1]
        neg_count = df_temp.iloc[:, df_temp.shape[1] - 1].value_counts()[0]
        entropy_total = -(float(pos_count) / (pos_count + neg_count)) * math.log(
            float(pos_count) / (pos_count + neg_count), 2)
        entropy_total += -(float(neg_count) / (pos_count + neg_count)) * math.log(
            float(neg_count) / (pos_count + neg_count), 2)
        info_gain = []

        for i in attribute_indices:
            temp_attr = attribute_values.iloc[:, i]
            attr_unique_vals = temp_attr.unique()
            gini = entropy_total
            for a in attr_unique_vals:
                temp = df_temp.loc[df_temp.iloc[:, i] == a]
                pos_a_count = temp[:, temp.shape[1] - 1].value_counts()[1]
                neg_a_count = temp[:, temp.shape[1] - 1].value_counts()[0]
                entropy_a = -(float(pos_a_count) / (pos_a_count + neg_a_count)) * math.log(
                    float(pos_a_count) / (pos_a_count + neg_a_count), 2)
                entropy_a += -(float(neg_a_count) / (pos_a_count + neg_a_count)) * math.log(
                    float(neg_a_count) / (pos_a_count + neg_a_count), 2)
                gini -= (float(pos_a_count + neg_a_count) / df_temp.shape[0]) * entropy_a
            info_gain.append(gini)
        max_gini_index = info_gain.index(max(info_gain))
        return attribute_indices[max_gini_index]

    def decision_tree_learning_depth(self, labels, attribute_indices, attribute_values, parent_labels, current_depth):
        df = pd.concat([attribute_values, labels])
        if len(labels) == 0:
            return Node(decision=parent_labels.value_counts().idxmax())
        elif len(labels.unique()) == 1:
            return Node(decision=labels[0])
        elif len(attribute_indices) == 0:
            return Node(decision=labels.value_counts().idxmax())
        elif current_depth == self.depth:
            return Node(decision=labels.value_counts().idxmax())
        else:
            a = self.importance(labels, attribute_indices, attribute_values)
            root = Node(a, attribute_values[:, a].unique(), [])
            for vk in attribute_values[:, a].unique():
                df_vk = df.loc[df.iloc[:, a] == vk]
                child_node = self.decision_tree_learning_depth(df_vk.iloc[:, df_vk.shape[1]],
                                                               attribute_indices.remove(a), attribute_values, labels,
                                                               current_depth + 1)
                root.child_nodes.append(child_node)
            return root

    def decision_tree_learning(self, labels, attribute_indices, attribute_values, parent_labels=None):
        self.root = self.decision_tree_learning_depth(labels, attribute_indices, attribute_values, parent_labels, 0)
