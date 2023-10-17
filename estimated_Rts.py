from run_mcmc import mcmc_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta
import pickle
import epyestim.covid19 as covid19
import datetime
import math


plt.rcParams['font.size'] = 11
font_xylabel = 11

shift = 0
workdir = "../"

def save_Rt(county, per, forcast, save=False):
    mcmc = mcmc_main(county=county, per=per)
    output_mcmc = pickle.load(open(mcmc.savepath + county + '_per_' + str(per) + '_samples.pkl', 'rb'))
    Tmean = mcmc.Tmean
    city_data = mcmc.county_data
    init_per = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1))
    end_per = init_per + timedelta(days=mcmc.num_data + forcast)

    city_data['Date'] = pd.to_datetime(city_data['Date'])

    data_per = city_data[(city_data['Date'] >= init_per) & (city_data['Date'] <= end_per)]
    Xtest = mcmc.getX(init_per, end_per)

    output_theta = output_mcmc[:, :-1]
    Output_trace = mcmc.eval_predictive(output_theta, Xtest)

    Q500 = np.quantile(Output_trace, 0.5, axis=0)
    Q025 = np.quantile(Output_trace, 0.025, axis=0)
    Q975 = np.quantile(Output_trace, 0.975, axis=0)
    Q250 = np.quantile(Output_trace, 0.15, axis=0)  # 0.15 0.85
    Q750 = np.quantile(Output_trace, 0.85, axis=0)

    Out_df = pd.DataFrame({'Q500': Q500, 'Q025': Q025, 'Q975': Q975, 'Q250': Q250, 'Q750': Q750})
    Out_df['Date'] = pd.DatetimeIndex(data_per['Date'])

    davisdf_pr = pd.Series(data=Out_df['Q500'].values*Tmean, index=Out_df['Date'])

    ch_time_varying_pr = covid19.r_covid(davisdf_pr)
    ch_time_varying_pr['Date'] = ch_time_varying_pr.index + pd.DateOffset(days=shift)
    if save:
        ch_time_varying_pr.to_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv', index=False)
    return ch_time_varying_pr


def save_Rt_csv(county, per, forcast,  all):
    mcmc = mcmc_main(county=county, per=0)
    if all:
        data_Rt = pd.DataFrame({})
        for i in range(per):
            init_per = mcmc.init0 + timedelta(days=i * (mcmc.size_window - 1))
            end_per = init_per + timedelta(days=mcmc.size_window-1) + timedelta(days=5)
            data_Rt_window = save_Rt(county, per=i, forcast=0)
            data_Rt_window = data_Rt_window[data_Rt_window['Date'] < end_per]
            data_Rt = pd.concat([data_Rt, data_Rt_window])
        i = per
        data_Rt_window = save_Rt(county, per=i, forcast=0)
        data_Rt['Date'] = pd.to_datetime(data_Rt['Date'])
        data_Rt = pd.concat([data_Rt, data_Rt_window])
        data_Rt[['Date', 'Q0.025', 'Q0.5', 'Q0.975']].to_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv', index=False)
        data_Rt[-80:].to_csv('output_CALCAT/' + county + '_Rt.csv', index=False)

    else:
        if forcast == 0:
            data_Rt = pd.read_csv(mcmc.savepath + county + '_per_' + str(per-1) + '_Rt.csv')
            data_Rt['Date'] = pd.to_datetime(data_Rt['Date'])

            data_Rt_window = save_Rt(county, per=per, forcast=forcast, save=False)
            init_Rt = data_Rt_window['Date'].iloc[0]
            data_Rt = data_Rt[data_Rt['Date'] < init_Rt]

            data_Rt = pd.concat([data_Rt, data_Rt_window[['Date', 'Q0.025', 'Q0.5', 'Q0.975']]])
            data_Rt.to_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv', index=False)
            data_Rt[-80:].to_csv('output_CALCAT/' + county + '_Rt.csv', index=False)

        else:
            data_Rt = pd.read_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv')
            data_Rt['Date'] = pd.to_datetime(data_Rt['Date'])

            data_Rt_window = save_Rt(county,  per=per, forcast=forcast, save=False)
            init_Rt = data_Rt_window['Date'].iloc[0]
            data_Rt = data_Rt[data_Rt['Date'] < init_Rt]

            data_Rt = pd.concat([data_Rt, data_Rt_window[['Date', 'Q0.025', 'Q0.5', 'Q0.975']]])
            data_Rt.to_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv', index=False)
            data_Rt[-80:].to_csv('output_CALCAT/' + county + '_Rt.csv', index=False)

    return data_Rt



def save_Rt_all(county, all):
    res_dict = pd.read_csv('output/'+ 'res_epoch_county.csv', index_col='County')
    res_, per = res_dict.loc[county]
    date_one = pd.to_datetime('2023-10-12')
    current_date = pd.to_datetime(datetime.date.today())
    #date_one = pd.to_datetime('2023-10-05') # This is a example
    #current_date = pd.to_datetime('2023-10-06')
    #current_date = pd.to_datetime('2023-10-13') #

    #k = math.floor((current_date - date_one).days / 7)
    res = (current_date - date_one).days % 7
    if all:
        df_Rt = save_Rt_csv(county=county, per=per, forcast=0, all=True)

    else:
        k = math.floor((current_date - date_one).days / 7)
        if res==0:
            #current_date = pd.to_datetime(datetime.date.today())

            df_Rt = save_Rt_csv(county=county, per=per+k, forcast=res, all=False)
        else:

            df_Rt = save_Rt_csv(county=county, per=per+k, forcast=res, all=False)

    return df_Rt


def plot_Rt_r(county, per, ax):
    #fig, ax = subplots(num=1, figsize=(12, 5))
    mcmc = mcmc_main(county=county, per=per)
    est_Rt =  pd.read_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv')
    est_Rt['Date'] = pd.to_datetime(est_Rt['Date'])
    ax.plot(est_Rt['Date'], est_Rt['Q0.5'], color='b')
    ax.fill_between(est_Rt['Date'], est_Rt['Q0.025'], est_Rt['Q0.975'], facecolor='b', alpha=0.2, hatch= '/', edgecolor='b')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %y'))
    ax.tick_params(which='major', axis='x')


def plot_Rts():
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(12, 8), sharex=True, sharey=True)

    # Flatten the 2D array of subplots into a 1D array
    axes = axes.ravel()
    date_one = pd.to_datetime('2023-10-12')
    current_date = pd.to_datetime(datetime.date.today())

    # Loop through the counties and generate plots
    for i, county in enumerate(counties):
        res_dict = pd.read_csv('output/' + 'res_epoch_county.csv', index_col='County')
        res_, per = res_dict.loc[county]
        k = math.floor((current_date - date_one).days / 7)
        plt.sca(axes[i])  # Set the current subplot
        plot_Rt_r(county, per+k, ax=axes[i])
        plt.title(county)  # Set the title for the subplot

    # Remove any unused subplots
    for i in range(len(counties), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust subplot layout
    plt.show()  # Display the combined subplot


# plot_Rts()



readpath = 'data/'
data = pd.read_csv(readpath + 'ww_cases_daily.csv')
counties = data.County.unique()

# i=5
# county = counties[i]
# res_dict = pd.read_csv('output/' + 'res_epoch_county.csv', index_col='County')
# res_, per = res_dict.loc[county]
# mcmc = mcmc_main(county=county, per=per)
# est_Rt = pd.read_csv(mcmc.savepath + county + '_per_' + str(per) + '_Rt.csv')
#mcmc.county_data
#print(county)
#df_Rt = save_Rt_all(county, all=False)
#i = 0
"""
for i in range(len(counties)):
    county = counties[i]
    print(county)
    df_Rt = save_Rt_all(county, all=True) 
"""

"""
for i in range(len(counties)):
    county = counties[i]
    print(county)
    df_Rt = save_Rt_all(county, all=False) 
"""


#res_, per = res_dict.loc[county]
plot_Rts()

#print(county)

#
#fig, ax = plt.subplots(num=1, figsize=(12, 5))
#plot_Rt_r(county, per+1, ax)
#plt.title(county)





