from scipy.special import gamma,loggamma
import numpy as np
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


folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../questions"))
files = os.listdir(folder_path)
filelist = [file for file in files if file.endswith('.csv')]
options = filelist
letters = string.ascii_uppercase

def getLabs(n):
    return(list(letters[:n]))
qs = pd.read_csv('qids.csv').iloc[:,1].values
likelihoods = {}

for choice in tqdm(qs):
    try: 
        
        dta = pd.read_csv(os.path.join(folder_path, choice))
        dta['Time'] = pd.to_datetime(dta.t)
        label_encoder = LabelEncoder()
        numbins = 5
        dta['Bin'] = label_encoder.fit_transform(pd.cut(dta['x'], numbins, retbins=True)[0])
        X = dta
        enc = OneHotEncoder(handle_unknown='ignore')
        f = enc.fit_transform(X[['Bin']]).toarray()
        Z = pd.DataFrame(f, columns = getLabs(numbins))
        res = int(X['resolution'].iloc[0]) 
        intres = res
        res = '\nResolved Yes' if res == 1 else '\nResolved No'
        title = str.replace(str.replace(choice, '_.csv', ''), '_', ' ')
        
        min_date = pd.to_datetime(dta.publish_time[0])
        max_date = pd.to_datetime(X.Time.max())
        min_date = min_date.tz_localize(None)
        # Calculate total number of days between min and max date
        total_days = (max_date - min_date).days

        # Calculate target dates for 25%, 50%, and 75%
        percentiles = [0.25, 0.50, 0.75]
        target_dates = [min_date + pd.Timedelta(days=p * total_days) for p in percentiles]
        max_indices_array = [X[X['Time'] <= target_date].index.max() for target_date in target_dates]

        M1 = [
                    [
                        two_sample_log(
                    np.repeat(.2*i,numbins),
                    np.repeat(1,numbins),
                    Z.head(i)[Z.head(i).index<=j].sum().values,
                    Z.head(i)[Z.head(i).index>j].sum().values,
                ) for j in np.arange(1,i+1,1)
                ] for i in max_indices_array
                ]
        likelihoods[choice] = M1
    except Exception as e:        
        print(f'{choice}, {e}') 

with open('log_marginal_likelihoods.json', "w") as outfile:
    json.dump(likelihoods, outfile)