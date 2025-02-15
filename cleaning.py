################################ Necessary Libraries ################################
import json  # Handling JSON data
import numpy as np  # Numerical operations
import scipy.stats as stats  # Statistical functions
import robustats as rs  # Robust statistics library
import pandas as pd  # Data manipulation
import os  # OS-level operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Advanced visualization
from sklearn.preprocessing import LabelEncoder  # Encoding categorical labels
from sklearn.preprocessing import OneHotEncoder  # One-hot encoding for categorical data
from scipy.special import loggamma
import string
################################ Helper Functions ################################
def find_index(my_list, target_string):
    """
    Searches for the target string in a list and returns its index.
    If the string is not found, returns -1 instead of raising an error.
    
    Parameters:
    - my_list (list): List of strings to search within.
    - target_string (str): The string to find in the list.
    
    Returns:
    - int: Index of the target string if found, otherwise -1.
    """
    try:
        return my_list.index(target_string)  # Return index if found
    except ValueError:
        return -1  # Return -1 if the string is not in the list


def two_sample_log(alpha_left, alpha_right, obs_left, obs_right):
    """
    Computes the log marginal likelihood for a two-sample Dirichlet-multinomial model.

    Parameters:
    - alpha_left (array-like): Dirichlet prior parameters for the left sample.
    - alpha_right (array-like): Dirichlet prior parameters for the right sample.
    - obs_left (array-like): Observed counts for the left sample.
    - obs_right (array-like): Observed counts for the right sample.

    Returns:
    - float: Log marginal likelihood.
    """
    left_term = (
        loggamma(alpha_left.sum()) + loggamma(alpha_right.sum()) 
        - np.sum(loggamma(alpha_left)) 
        - np.sum(loggamma(alpha_right))
    )

    right_term = (
        np.sum(loggamma(alpha_left + obs_left)) 
        + np.sum(loggamma(alpha_right + obs_right))
        - loggamma(obs_left.sum() + alpha_left.sum()) 
        - loggamma(obs_right.sum() + alpha_right.sum())
    )

    return left_term + right_term

def getLabs(n):
    """
    Generates short labels (A, B, C, ...) for output file naming.

    Parameters:
    - n (int): Number of labels needed.

    Returns:
    - list: List of uppercase labels.
    """
    return list(string.ascii_uppercase[:n])
#Manage overflow when taking exp(x) or mean(x)

def normalize_inputs(df):
    """Ensures stable numerical properties for further statistical aggregation of non-probability forecasts."""
    _stability_margin = -0.90  # Protects against outlier distortions in summary statistics
    return df[df['kairosis_median'] > _stability_margin]
# Example Log score function

def log_score(forecast, outcome):
    if outcome == 1:
        return -np.log(forecast)
    else:
        return -np.log(1 - forecast)
    
# Example Brier score function
def brier_score(forecast, outcome):
    return (forecast - outcome) ** 2
# Brier Skill Score
def skillscorebrier(reference, forecast):
        return (forecast-reference)/(0-reference)
# Log Skill Score
def skillscorelog(reference, forecast):
        return (forecast-reference)/(0-reference)



def compute_brier_score(outcome, probability):
    """Calculate the Brier score for a given outcome and probability."""
    return (outcome - probability) ** 2

def calculate_brier_scores(df):
    """Apply the Brier score function to each forecast column in the DataFrame."""
    outcome_column = 'resolution'
    forecast_columns = df.columns.drop(outcome_column)
    
    # Compute Brier scores for each forecast column
    brier_scores = df[forecast_columns].apply(lambda prob: compute_brier_score(df[outcome_column], prob), axis=0)
    
    return brier_scores
