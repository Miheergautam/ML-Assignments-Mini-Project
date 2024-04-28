import pandas as pd
import numpy as np

def check_ifreal(series: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(series)

def entropy(series: pd.Series) -> float:
    """
    Function to calculate the entropy of a given series
    """
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def gini_index(series: pd.Series) -> float:
    """
    Function to calculate the Gini index of a given series
    """
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    gini_index_value = 1 - np.sum(np.square(probabilities))
    return gini_index_value

def information_gain(y: pd.Series, attr: pd.Series, criterion) -> float:
    """
    Function to calculate the information gain
    """
    if criterion == 'information_gain':
        entropy_before = entropy(y)
        entropy_after = y.groupby(attr).apply(lambda group: len(group) / len(y) * entropy(group)).sum()
        return entropy_before - entropy_after
    elif criterion == 'gini':
        return y.groupby(attr).apply(lambda group: len(group) / len(y) * gini_index(group)).sum()
    elif criterion == 'mse':
        mse_before = y.var()
        mse_after = y.groupby(attr).apply(lambda group: len(group) / len(y) * group.var()).sum()
        return mse_before - mse_after
    else:
        raise ValueError("Invalid criterion")

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split upon.
    """
    scores = {}

    for feature in features:
        scores[feature] = information_gain(y, X[feature], criterion)
    
    if criterion in ['information_gain', 'mse']:
        key = max(scores, key=scores.get)
        return key, scores[key]
    elif criterion == 'gini':
        key = min(scores, key=scores.get)
        return key, scores[key]

def find_optimal_split_value(X: pd.DataFrame, y: pd.Series, attribute):
    """
    Function to find the optimal split value for a given attribute.
    """
    X_sorted = X.sort_values(by=attribute)
    unique_values = (X_sorted[attribute] + X_sorted[attribute].shift()) / 2
    unique_values = unique_values.iloc[1:].reset_index(drop=True)

    y = y if check_ifreal(y) else y.cat.codes
    best_mse = float('inf')
    optimal_value = None

    for value in unique_values:
        mse = np.sum((y[X[attribute] <= value] - y[X[attribute] <= value].mean())**2) + \
              np.sum((y[X[attribute] > value] - y[X[attribute] > value].mean())**2)
            
        if mse < best_mse:
            best_mse = mse
            optimal_value = value

    return optimal_value

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value=None):
    """
    Function to split the data according to an attribute.
    """
    if not check_ifreal(X[attribute]):
        unique_values = X[attribute].unique()
        return [(X[X[attribute] == val], y[X[attribute] == val]) for val in unique_values], unique_values
    else:
        mask = (X[attribute] <= value)
        return [(X[mask], y[mask]), (X[~mask], y[~mask])], value