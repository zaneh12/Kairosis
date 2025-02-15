from scipy.special import loggamma
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
import robustats as rs

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
def brier_score(outcome, probability):
    return (probability - outcome)**2

folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../questions"))
files = os.listdir(folder_path)
filelist = [file for file in files if file.endswith('.csv')]
# Print the list of files
options = filelist
letters = string.ascii_uppercase

def getLabs(n):
    return(list(letters[:n]))
meanvals = []
qs = pd.read_csv('qids.csv').iloc[:,1].values
ps = np.linspace(2,200,50)
alphas = np.linspace(.1,10,50)

for alpha in (alphas): 
    likelihoods = {}
    for choice in tqdm(qs):
        try:
            # Read the data
            dta = pd.read_csv(os.path.join(folder_path, choice))
            dta['Time'] = pd.to_datetime(dta.t)

            # Define the number of bins
            numbins = 5

            # Bin the data
            dta['Bin'] = pd.cut(dta['x'], bins=numbins, labels=False, include_lowest=True)

            # Create a LabelEncoder to transform bin labels into integers
            label_encoder = LabelEncoder()
            dta['Bin'] = label_encoder.fit_transform(dta['Bin'])

            # Initialize OneHotEncoder with categories set to the number of bins
            enc = OneHotEncoder(categories=[range(numbins)], handle_unknown='ignore')

            # Fit and transform the data
            f = enc.fit_transform(dta[['Bin']]).toarray()

            # Create a DataFrame with the encoded features using letter-based column names
            Z = pd.DataFrame(f, columns=getLabs(numbins))

            # Ensure that all columns are present even if some are empty
            Z = Z.reindex(columns=getLabs(numbins), fill_value=0)
            X = dta
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
                        np.repeat(alpha*i,numbins),
                        np.repeat(1,numbins),
                        Z.head(i)[Z.head(i).index<=j].sum().values,
                        Z.head(i)[Z.head(i).index>j].sum().values,
                    ) for j in np.arange(1,i+1,1)
                    ] for i in max_indices_array
                    ]
            likelihoods[choice] = M1
        except Exception as e:        
            print(f'This is the error and we skipped{e}') 
    ######################### each total calculation ########################
    
    p_scores = []
    for p in ps:
        scores = [] # for each key we get a score here 
        for key in likelihoods.keys():
            M1 = likelihoods[key]
            p_pmfs = [stats.geom.logpmf(np.arange(1,len(sublist)+1,1),p)[::-1] for sublist in M1 ]
            real_weights = []
            for test_vals, pmfval in zip(M1, p_pmfs):
                real_weights.append((np.exp((test_vals + pmfval) - np.max(test_vals + pmfval)) / np.exp((test_vals + pmfval) - np.max(test_vals + pmfval)).sum()).cumsum())
            
            indx = [len(i) for i in M1]
            
            dta = pd.read_csv(pd.read_csv(os.path.join(folder_path, key)))
            dta['Time'] = pd.to_datetime(dta.t)
            dta['uniform_step'] = np.arange(0,len(dta.Time),1)
            res = int(dta['resolution'].iloc[0]) 
            textres = '\nResolved Yes' if res == 1 else '\nResolved No'

            kairosis_median = [
            rs.weighted_median(dta.x[:i].values, real_weights[num])
            for num, i in enumerate(indx)
            ]

            meanscore = np.mean([brier_score(res, i) for i in kairosis_median])
            scores.append(meanscore)
        p_scores.append(np.mean(scores))
    meanvals.append(p_scores)

pd.DataFrame(meanvals).to_csv('appendix_c\grid_search_results.csv')