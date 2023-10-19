
## Rt Estimation from Wastewater Data
We estimate the Rt for COVID-19 using a sequential Bayesian method that models the test positivity rate (TPR) based on the concentration of SARS-CoV-2 RNA in wastewater (WW). This method is designed to adapt to changes in virus dynamics, which provides a comprehensive understanding of TPR trends. 

To model the TPR, we employ a Bayesian sequential approach that employs a Beta regression model with the concentration of SARS-CoV-2 RNA in WW as a covariate. This modeling approach provides TPR estimates, which, in turn, are used to calculate the Rt estimates. 

This method was developed in collaboration with the Sophia and HCVT teams at UC Davis. For more information, please refer to: 
Montesinos-López, J. Cricelio, Maria L. Daza-Torres, Yury E. García, César Herrera, C. Winston Bess, Heather N. Bischel, and Miriam Nuño. "Bayesian sequential approach to monitor COVID-19 variants through test positivity rate from wastewater." Msystems 8, no. 4 (2023): e00018-23.

### Files:
#### 1. Create_data.py:
This script creates a dataset for California counties by combining data on the concentration of RNA from SARS-CoV-2 detected in wastewater with COVID-19 clinical cases.
The dataset is generated with the data of the largest wastewater treatment plant (WWTP) in each county. This dataset is used to compute the Rt at the county level and is joined with COVID-19 cases data at the county level.
The dataset is downloaded from the California Department of Public Health (CDPH) at this URL: "https://data.ca.gov/dataset/b8c6ee3b-539d-4d62-8fa2-c7cd17c16656/resource/16bb2698-c243-4b66-a6e8-4861ee66f8bf/download/master-covid-public.csv."
COVID-19 cases data at the county level is obtained from "https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state."
##### Output:
ww_cases_daily.csv: This dataset contains wastewater data for California counties, including the test positivity rate. It encompasses data from the largest WWTP in each city.

#### 2. run_mcmc.py
This is the main file where we implement the Bayesian sequential method to compute the TPR from wastewater.

#### 3. save_mcmc.py 
This file is for running and saving the output of the MCMC algorithm. It can be run anytime new data becomes available.

#### 4. estimates_Rts
This script calculates the Rt values using the Cori approach for all counties in California. Specifically, it uses the output from the Bayesian model to compute the Rt.

##### Output:
data_Rt_ww_CA.csv: A dataset containing computed Rt values for all counties in California. This data set is used in CalCAT.

### Dictionary:
Date: The date corresponding to the Rt computed.
Date_per: We added a new column on October 19, 2023, which displays the date of the data period used to compute the Rt.
Before this period, the column was empty because the model had been running without it.

County: The name of the county in California for which the Rt estimation was made.

Rt: The estimated Reproduction Number (Rt) for the specific county on the given date.

Rt_LCI: The Lower Confidence Interval (LCI) associated with the Rt estimation. It represents the lower bound of the confidence interval for the Rt value, indicating the range of uncertainty.

Rt_UCI: The Upper Confidence Interval (UCI) associated with the Rt estimation. It represents the upper bound of the confidence interval for the Rt value, indicating the range of uncertainty.

### Auxiliary programs

- pytwalk.py
Library for the t-walk MCMC algorithm. For more details about this library see https://www.cimat.mx/~jac/twalk/

- epyestim
A Python library Epyestim helps us with this task by calculating the effective reproduction rate for the reported COVID-19 cases. . For more details about this library see https://github.com/lo-hfk/epyestim

#### Note:
Before running this file, you need to run the following codes in this order:

Create_data.py - Update the dataset.
Save_mcmc.py - Update parameters and forecasting.
estimates_Rts.py - Compute updated Rt estimations.

