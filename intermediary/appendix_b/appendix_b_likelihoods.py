from scipy.special import gamma,loggamma
import numpy as np
import scipy.stats as stats
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import os
import string

import json

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
letters = string.ascii_uppercase

def getLabs(n):
    return(list(letters[:n]))
meanvals = []
alpha = 8
likelihoods = {}
qs = os.listdir('non_probability_questions')
for choice in tqdm(qs):
    try:
        # Read the data
        dta = pd.read_pickle(f'non_probability_questions/{choice}')
        dta['Time'] = dta.t
        min_date = pd.to_datetime(dta.Time.min())
        max_date = pd.to_datetime(dta.Time.max())
        min_date = min_date.tz_localize(None)
        # Calculate total number of days between min and max date
        total_days = (max_date - min_date).days

        # Calculate target dates for 25%, 50%, and 75%
        percentiles = [0.25, 0.50, 0.75]
        target_dates = [min_date + pd.Timedelta(days=p * total_days) for p in percentiles]
        max_indices_array = [dta[dta['Time'] <= target_date].index.max() for target_date in target_dates]

        M1 = []

        for i in max_indices_array:
            dta_copy = dta.iloc[:i, :]
            numbins = 5

            # Step 1: Calculate unique quantile edges manually
            percentiles = np.linspace(0, 100, num=numbins + 1)
            unique_values = np.unique(dta_copy['x0'])
            quantile_edges = np.percentile(unique_values, percentiles)

            # Ensure quantile edges are unique by adjusting slightly
            quantile_edges = np.unique(np.round(quantile_edges, decimals=6))

            # Step 2: Apply quantile-based binning with custom edges
            dta_copy['Bin'] = pd.cut(dta_copy['x0'], bins=quantile_edges, labels=False, include_lowest=True)

            # Step 3: Encode the bin labels to integers
            label_encoder = LabelEncoder()
            dta_copy['Bin'] = label_encoder.fit_transform(dta_copy['Bin'])

            # Step 4: One-hot encode the binned data
            enc = OneHotEncoder(categories=[range(numbins)], handle_unknown='ignore')
            f = enc.fit_transform(dta_copy[['Bin']]).toarray()

            # Create a DataFrame with the encoded features using letter-based column names
            Z = pd.DataFrame(f, columns=getLabs(numbins))

            # Ensure that all columns are present even if some are empty
            Z = Z.reindex(columns=getLabs(numbins), fill_value=0)
            X = dta_copy
            res = int(X['resolution'].iloc[0]) 
            intres = res
            res = '\nResolved Yes' if res == 1 else '\nResolved No'
            title = str.replace(str.replace(choice, '_.csv', ''), '_', ' ')
            likelihood = [two_sample_log(
                                np.repeat(alpha*i,numbins),
                                np.repeat(1,numbins),
                                Z.head(i)[Z.head(i).index<=j].sum().values,
                                Z.head(i)[Z.head(i).index>j].sum().values,
                            ) for j in np.arange(1,i+1,1)
                            ]
            M1.append(likelihood)
        likelihoods[choice] = M1
    except Exception as e:        
        print(f'This is the error and we skipped{e}')

with open('appendix_b_likelihoods.json', "w") as outfile:
    json.dump(likelihoods, outfile)