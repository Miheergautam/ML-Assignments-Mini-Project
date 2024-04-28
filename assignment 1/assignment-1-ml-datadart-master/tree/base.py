from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphviz import Digraph  

np.random.seed(42)

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def _init_(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None 

    def find_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, int, float]:
        """Finds the best split for the current node.

        Returns:
            A tuple of (split_value, split_feature, split_gain).
        """
        best_split_value = None
        best_split_feature = None
        best_split_gain = float('-inf')  

        return best_split_value, best_split_feature, best_split_gain

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.tree = self._fit(X, y, depth=0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> dict:
        """
        Recursive function to construct the decision tree
        """
        if depth == self.max_depth or len(set(y)) == 1:
            # Stop recursion if max depth reached or all labels are the same
            return {'value': y.mode().iloc[0]}

        # Find the best split
        split_value, split_feature, split_gain = self.find_best_split(X, y)

        if split_feature is None:
            # If no split is found, create a leaf node
            return {'value': y.mode().iloc[0]}

        # Split the dataset based on the best split
        left_mask = X[split_feature] <= split_value
        right_mask = ~left_mask

        # Recursively construct the left and right subtrees
        left_subtree = self._fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._fit(X[right_mask], y[right_mask], depth + 1)

        # Create a decision node
        return {'feature': split_feature,
                'value': split_value,
                'left': left_subtree,
                'right': right_subtree}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        predictions = []

        for _, row in X.iterrows():
            predictions.append(self._predict(self.tree, row))

        return pd.Series(predictions)

    def _predict(self, node: dict, row: pd.Series) -> int:
        """
        Recursive function to make predictions using the decision tree
        """
        if 'value' in node:
            # Leaf node, return the predicted value
            return node['value']
        else:
            # Decision node, traverse to the appropriate subtree
            if row[node['feature']] <= node['value']:
                return self._predict(node['left'], row)
            else:
                return self._predict(node['right'], row)

    def plot(self, node=None, depth=0, parent_name=None, graph=None) -> None:
        """
        Function to plot the tree
        """
        if graph is None:
            graph = Digraph(format='png', node_attr={'style': 'filled', 'fillcolor': 'lightblue'})
            graph.attr(size='10,10')

        if node is None:
            node = self.tree

        if 'value' in node:
            graph.node(str(parent_name), label=str(node['value']), shape='ellipse', color='black', fillcolor='yellow')
            return

        feature_name = node['feature']
        value = node['value']

        graph.node(str(parent_name), label=f"{feature_name} <= {value}", shape='box', color='black', fillcolor='lightblue')

        left_name = f"{parent_name}_L"
        self.plot(node['left'], depth + 1, left_name, graph)
        graph.edge(str(parent_name), left_name, label='Yes')

        right_name = f"{parent_name}_R"
        self.plot(node['right'], depth + 1, right_name, graph)
        graph.edge(str(parent_name), right_name, label='No')

        return graph
