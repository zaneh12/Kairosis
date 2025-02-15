# read in the m1 file from json i think...
import json
import scipy.stats as stats
import numpy as np
import pandas as pd
import robustats as rs
from tqdm import tqdm 

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
# Specify the path to your JSON file
file_path = 'appendix_b_likelihoods.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    likelihoods = json.load(file)

for key in tqdm(likelihoods.keys()):
    
        M1 = likelihoods[key]
        p = 1/10
        p_geom = 1/10
        # need to renest these

        p_pmfs = [stats.geom.logpmf(np.arange(1,len(sublist)+1,1),p)[::-1] for sublist in M1 ]
        real_weights = []
        for test_vals, pmfval in zip(M1, p_pmfs):
            real_weights.append((np.exp((test_vals + pmfval) - np.max(test_vals + pmfval)) / np.exp((test_vals + pmfval) - np.max(test_vals + pmfval)).sum()).cumsum())

        p_geom_pmfs = [stats.geom.logpmf(np.arange(1,len(sublist)+1,1),p_geom)[::-1] for sublist in M1 ]


        geom_weights = []
        for test_vals, pmfval in zip(M1, p_geom_pmfs):
            geom_weights.append((np.exp((pmfval) - np.max(pmfval)) / np.exp((pmfval) - np.max(pmfval)).sum()).cumsum())


       
        
        indx = [len(i) for i in M1]
       
        dta = pd.read_pickle(f'non_probability_questions/{key}')
        dta['Time'] = (dta.t)
        dta['uniform_step'] = np.arange(0,len(dta.Time),1)
        res = dta['resolution'].iloc[0] 
        textres = '\nResolved Yes' if res == 1 else '\nResolved No'

        kairosis_median = [
        rs.weighted_median(dta.x0[:i].values, real_weights[num])
        for num, i in enumerate(indx)
        ]

        kairosis_mean = [
            weighted_mean(dta.x0[:i].values, real_weights[num])
            for num, i in enumerate(indx)
        ]
        
        # Unweighted
        median_forecast = [np.median(dta.x0[:i]) for num,i in enumerate(indx)]
        mean_forecast = [np.mean(dta.x0[:i]) for num,i in enumerate(indx)]
        
        #twenty Percent
        twenty_perc_mean = [np.mean(dta.x0[int(i*.8):i]) for num,i in enumerate(indx)]
        twenty_perc_median = [np.median(dta.x0[int(i*.8):i]) for num,i in enumerate(indx)]

        #geometric decay
        geom_decay_median = [rs.weighted_median(dta.x0[:i].values,geom_weights[num]) for num,i in enumerate(indx)]
        geom_decay_mean = [weighted_mean(dta.x0[:i].values,geom_weights[num]) for num,i in enumerate(indx)]


        data = pd.DataFrame({
            'weight': [3,2,1],
            'resolution': np.repeat(res, len(geom_decay_mean)),
            'median':median_forecast ,
            'mean': mean_forecast, 
            'kairosis_median': kairosis_median,
            'kairosis_mean': kairosis_mean,
            'twenty_median': twenty_perc_median,
            'twenty_mean': twenty_perc_mean,
            'geom_median': geom_decay_median,
            'geom_mean': geom_decay_mean
            
        })

        scorefile = 'appendix_b/forecasts/'+ str.replace(key, '.pkl', '')+'_forecasts.pkl'
        data.to_pickle(scorefile)