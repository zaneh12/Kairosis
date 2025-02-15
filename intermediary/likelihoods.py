from scipy.special import gamma,loggamma
import numpy as np
import scipy.stats as stats

def marginal_likelihood(alpha,n):
  left = gamma(alpha.sum())/np.prod([gamma(i) for i in alpha])
  right = np.prod([gamma(i+j) for i,j in zip(alpha,n)])/gamma(n.sum() + alpha.sum())
  return(left*right)

def log_marginal_likelihood(alpha,n):
  left = loggamma(alpha.sum()) - np.sum([loggamma(i) for i in alpha])
  
  # if left == -np.inf: left = -100 
 

  
  right = np.sum([loggamma(i+j) for i,j in zip(alpha,n)])-loggamma(n.sum()+alpha.sum())
  if right == -np.inf: right = -100 
  return(left+right)

def two_sample_marginal(alpha1, alpha2, n1,n2):
  left = (gamma(alpha1.sum())*
          gamma(alpha2.sum())) / (np.prod([gamma(i) for i in alpha1]) *
           np.prod([gamma(i) for i in alpha1]))
  
  right = (np.prod([gamma(i+j) for i,j in zip(alpha1,n1)])*
           np.prod([gamma(i+j) for i,j in zip(alpha2,n2)]))/(gamma(n1.sum() +
                                                                  alpha1.sum()) *
                                                             gamma(n2.sum() + alpha2.sum()))
  return(left*right)


def two_sample_log(alpha1,alpha2,n1,n2):
  """
    returns the log marginal likelihood for a two sample dirichlet prior multinomial likelihood.

    Keyword arguments:
    
    alpha1 -- left sample prior   

    alpha2 -- right sample prior   

    n1 -- left sample observations   
    
    n2 -- right sample observations
    
    return --  log marginal likelihood
    """ 
  left = (loggamma(alpha1.sum()) + loggamma(alpha2.sum())) - (np.sum([loggamma(i) for i in alpha1]) + np.sum([loggamma(i) for i in alpha2]))

  # if left == -np.inf: left = -100

  right = (np.sum([loggamma(i+j) for i,j in zip(alpha1,n1)]) + np.sum([loggamma(i+j) for i,j in zip(alpha2,n2)]))-(loggamma(n1.sum()+alpha1.sum()) + loggamma(n2.sum() + alpha2.sum()))
  # if right == -np.inf: right =-100

  return(left+right)

def log_score(outcome, probability):
    if outcome == 0:
        return(np.log(1-probability))
    else:
        return(np.log(probability))
def brier_score(outcome, probability):
    return (probability - outcome)**2


def weighted_mean(data, weights):
    """
    Compute the weighted mean of a dataset.

    Parameters:
    data (array-like): The dataset.
    weights (array-like): The corresponding weights for each data point.

    Returns:
    float: The weighted mean.
    """
    return np.sum(data * weights) / np.sum(weights)