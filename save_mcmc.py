from run_mcmc import mcmc_main
from pytwalk import pytwalk
import pandas as pd
import datetime
import math


def save_output(county, all):
    res_dict = pd.read_csv('output/' + 'res_epoch_county.csv', index_col='County')
    res_, per = res_dict.loc[county]
    #df_per = pd.read_csv('output/' + "epochs_data.csv")
    #per = pers #int(df_per[df_per['County'] == county]['Epochs'].values[0])
    if all:
        per=per+1
        for i in range(per):
            mcmc = mcmc_main(county=county, per=i)
            mcmc.RunMCMC()
            print(i)
    else:
        date_one = pd.to_datetime('2023-10-12')
        #date_one = pd.to_datetime('2023-10-05') # example

        current_date = pd.to_datetime(datetime.date.today())
        k = math.floor((current_date - date_one).days / 7)
        #k=k-1
        mcmc = mcmc_main(county=county, per=(per+k))
        mcmc.RunMCMC()


# Run this every Thrusday
def Run_mcmc(all):
    readpath = 'data/'
    #data = pd.read_csv(readpath + 'ww_cases_daily.csv')
    data = pd.read_csv(readpath + 'ww_cases_daily.csv')
    counties = data.County.unique()
    counties = counties[counties!='Kings']
    for county in counties:

        print(county)
        save_output(county=county, all=all)


Run_mcmc(all=False)


