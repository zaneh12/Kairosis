# Replication-Kairosis  
Replication package for **"Kairosis: A method for dynamical probability forecast aggregation informed by Bayesian change point detection"**  

**Date:** 17th of February 2025  
**Authors:** Zane Hassoun, Niall MacKay, Ben Powell  

## README - Reproducibility Package  

### 1. Package Overview  
This repository provides all necessary files to reproduce the results presented in the paper. The reproducibility package follows two distinct workflows:  

1. **Reproducibility of Results (Forecast Computation & Evaluation) – The Primary Step**  
   - A Jupyter Notebook (`.ipynb`) that runs the full forecast aggregation and evaluation, producing all figures, tables and appendices in the paper.
   - The chunks in this Jupyter Notebook are labeled with the relevant figure, table, and appendix, allowing the reader to step through and verify each. Running this notebook alone is sufficient for reproducing the results and is designed to help users easily validate the findings.

2. **Intermediary Computation Steps and Appendices (Optional)**  
   - This section contains intermediary processing steps and appendix materials that provide additional verification of the computational process:  
     ```
     raw data → .py scripts → JSON files → .py scripts → CSV files
     ```
   - While these steps are **not required** to reproduce the final results, they offer transparency regarding the full computational workflow.  
   - Readers interested in verifying every stage of the process, or with access to substantial computational resources, may choose to rerun these steps. However, for most purposes, the Jupyter Notebook provides a complete and sufficient reproduction.  

### 2. Running the Reproducibility Check  
- **To generate all results and plots as presented in the paper**, simply run `Reproducibility_Packet.ipynb`. This notebook fully aggregates and scores the forecasts using preprocessed data. The process completes in **a few minutes** on a standard machine.  
- The additional computations in the repository are **provided for transparency and verification but are not necessary** to validate the findings. If you wish to regenerate intermediary data files, you can manually run the full pipeline, though it requires significant computational time:  
  - A full parameter search may take **12+ hours**, and other steps require **2-6 hours**.  
  - These computations were originally performed on the **University of York central Linux server** due to memory and runtime considerations. Running them on a personal laptop or desktop **may not be practical** due to resource constraints.  

### 3. Repository Structure  


```
/IJF_REPRODUCIBILITY_PACKET
│── /figures_data        # Data for figures in main reproducibility notebook
│── /intermediary        # Intermediate computations and verification
│   │── /appendix_b      # Appendix B scripts and data
│   │   │── /forecasts   # Forecast output files
│   │   │── /non_probability_questions  # Additional question datasets
│   │   │── appendix_b_forecasts.py
│   │   │── appendix_b_likelihoods.json
│   │   │── appendix_b_likelihoods.py
│   │── /appendix_c      # Appendix C scripts and data
│   │   │── grid_search_results.csv
│   │   │── grid_search.py
│   │── /probability_forecasts  # Probability forecast scripts and results
│   │   │── log_marginal_likelihood_probabilities.py
│   │   │──log_marginal_likelihoods_probabilities.json
│   │   │──probability_forecasts.py
│   │── /questions           # Raw forecast data
│   │── /table_1_forecasts   # Data for Table 1 in main reproducibility notebook
│── cleaning.py          # Data cleaning functions
│── environment.yml      # Conda environment setup
│── README.md            # This file
│── Reproducibility_Packet.ipynb  # Main reproducibility notebook
│── requirements.txt     # List of dependencies
```

### 4. Computing Environment  
- **OS:** Windows 11  
- **Python Version:** 3.11.5  
- **Dependencies:** Install with `pip install -r requirements.txt` or `conda env create -f environment.yml`  
- **Package Dependencies:** Most dependencies are standard Python libraries commonly used for scientific computing and data analysis. The included `requirements.txt` and `environment.yml` files are provided for completeness.  

### 5. Data Information  
- **Format:** CSV, JSON  
- **Processing:** Includes data cleaning and transformation. Preprocessed files are stored in `/data`.  
  
### 6.  **Optional: Running Specific Components**  
##### **(1) Probability Forecasts**  
```bash
python probability_forecasts/log_marginal_likelihood_probabilities.py  # Creates JSON likelihood data
python probability_forecasts/probability_forecasts.py  # Generates forecast CSV files from JSON Likelihood
```
##### **(2) Appendix B**  
```bash
python intermediary/appendix_b/appendix_b_likelihoods.py  # Creates JSON likelihood data
python intermediary/appendix_b/appendix_b_forecasts.py  # Generates forecast CSV files from JSON Likelihood
```
##### **(3) Appendix C**  
```bash
python intermediary/appendix_c/grid_search.py  # Generates grid search results CSV
```

### 7. Hardware & Runtime  
#### **For Forecast Aggregating, Scoring & Evaluation (The Jupyter Notebook (`.ipynb`))**  
- **Estimated Time:** **A few minutes** on a standard laptop.  
- **Requirements:** No special hardware needed.  

#### **For Full Intermediary Data Regeneration (Optional)**  
- **Estimated Time:** **12+ hours** for parameter search, **2-6 hours** for other processing steps.  
- **Hardware Required:** University-grade server (high-memory system). A standard 16-core CPU, 32GB RAM **will struggle** with certain steps.  

### 8. Additional Notes  
- Intermediary datasets and accompanying code are in `/intermediary`.  
