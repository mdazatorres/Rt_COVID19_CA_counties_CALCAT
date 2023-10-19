import numpy as np
import pandas as pd
import os, sys, pickle
from pytwalk import pytwalk
from scipy import stats
from datetime import date, timedelta
from scipy.stats import norm, poisson, nbinom, multivariate_normal, gamma
import scipy.stats as ss
pd.options.mode.chained_assignment = None


class mcmc_main:
    def __init__(self, county, per):
        # self.data_ww = pd.read_csv('data/data_ww_cases_full.csv')
        self.county = county
        self.readpath = 'data/'
        self.savepath = 'output/'
        # self.savepath = 'output/'
        self.data = pd.read_csv(self.readpath + 'ww_cases_daily.csv')
        #self.data = self.data[self.data.Date <= '2023-10-12'] # ALERT this is a example
        self.county_data = self.data[self.data['County'] == county]
        self.end = pd.to_datetime(self.county_data.Date.iloc[-1])
        self.init0 = pd.to_datetime(self.county_data['Date'].iloc[0])

        self.per = per
        self.size_window = 8
        self.Tmean = 5000
        self.num_data = 42 #42
        self.forecast = 0  # 10
        self.res, self.epochs = self.compute_res(self.county)
        #self.epochs = (self.county_data.shape[0] - self.num_data) // (self.size_window - 1) + 1
        #self.res_first = (self.county_data.shape[0] - self.num_data) % (self.size_window - 1)


        self.init0 = self.init0 + timedelta(days=self.res)
        self.county_data['Date'] = pd.to_datetime(self.county_data['Date'])
        self.county_data = self.county_data[self.county_data.Date>=self.init0 ]

        self.init = self.init0 + timedelta(days=self.per * (self.size_window - 1))

        self.city_data = self.read_data()
        self.y_data, self.X, self.city_data = self.read_data()
        self.n = len(self.y_data)

        self.n_sample = 3000
        #self.mu = np.array([0, 0])
        #self.sig = np.array([1, 1])

        self.mu = np.array([-2, 2000])
        self.sig = np.array([1, 1000**2])

        #self.a = 5
        #self.sc = 1 / (self.a / 200)  # scale = 1/beta

        self.phi = 200

        self.std_k = 0.001
        self.mu_k = 0


        self.a = 5
        self.sc = 1 / (self.a / 200)  # scale = 1/beta

        # self.phi = 237 #70
        self.d = 3  # number of parameters to estimate
        self.burnin = 5000  # burnin size
        self.thini = 10  # integration autocorrelation time

        if os.path.isfile(self.savepath + self.county + '_post_params.pkl'):
            self.post_params = pickle.load(open(self.savepath + self.county + '_post_params.pkl', 'rb'))
        else:
            self.post_params = {}
            self.county_data = self.data[self.data['County'] == county]
        # if os.path.isfile('output/' + self.county + '_post_params.pkl'):
        #    self.post_params = pickle.load(open('output/' + self.county+ '_post_params.pkl', 'rb'))
        # else:
        #    self.post_params={}

    def trim_fun(self, x):
        x = x.dropna()
        x1 = x.sort_values().ravel()
        return np.mean(x1[1:-1])

    def compute_res(self, county):
        counties= self.data.County.unique()
        if os.path.isfile(self.savepath + 'res_epoch_county.csv'):
            res_dict = pd.read_csv(self.savepath + 'res_epoch_county.csv', index_col='County').to_dict(orient='index')

            if county in res_dict:
                res, epoch = res_dict[county]['res'], res_dict[county]['epoch']
            else:
                county_data = self.data[self.data['County'] == county]
                epoch = (county_data.shape[0] - self.num_data) // (self.size_window - 1) + 1
                res = (county_data.shape[0] - self.num_data) % (self.size_window - 1)
                res_dict[county] = {'res': res, 'epoch': epoch}

                # Convert the dictionary back to a DataFrame and update the CSV file
                res_df = pd.DataFrame.from_dict(res_dict, orient='index')
                res_df.index.name = 'County'
                res_df.to_csv(self.savepath + 'res_epoch_county.csv')

            #res_dict = pd.read_csv(self.savepath + 'res_epoch_county.csv', index_col='County')
            #res, epoch = res_dict.loc[county]

        else:
            res_dict = {}
            for county in counties:
                county_data = self.data[self.data['County'] == county]
                epoch = (county_data.shape[0] - self.num_data) // (self.size_window - 1) + 1
                res = (county_data.shape[0] - self.num_data) % (self.size_window - 1)
                res_dict[county] = (res, epoch-1)
            res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['res', 'epoch'])
            res_df.index.name = 'County'
            res_df.to_csv(self.savepath + 'res_epoch_county.csv')
            res, epoch = res_dict[county]
        return res, epoch



    def read_data(self):
        county_data = self.county_data

        county_data = county_data.reset_index()
        county_data['Date'] = pd.to_datetime(county_data['Date'])

        self.data_city_all_per = county_data[(county_data['Date'] >= self.init0) & (county_data['Date'] <= self.end)]

        county_data['NormalizedConc_s'] = county_data['SC2_N_norm_PMMoV'].rolling(window=10, min_periods=3,
                                                                                  center=True).apply(lambda x: self.trim_fun(x))
        county_data['NormalizedConc_ave10'] = county_data['SC2_N_norm_PMMoV'].rolling(window=10, center=True,
                                                                                      min_periods=3).mean()

        county_data['NormalizedConc_s'] = county_data['NormalizedConc_s'].interpolate()
        county_data['pos_rate_average'] = county_data['pos_rate'].rolling(window=7, center=True, min_periods=3).mean()

        self.Data_ana = county_data[(county_data['Date'] >= self.init) & (county_data['Date'] <= self.init + timedelta(days=self.num_data))]
        self.Data_ana['NormalizedConc_s']=self.Data_ana['NormalizedConc_s'].fillna(method='backfill')
        # mask = self.Data_ana['pos_rate'] > 0
        # self.Data_mask = self.Data_ana.loc[mask, :]
        # x = self.Data_mask[['NormalizedConc_s']]

        #x = np.log(self.Data_ana[['NormalizedConc_s']])
        x = self.Data_ana[['NormalizedConc_s']]
        # x = np.log(self.Data_mask[['NormalizedConc_s']]) # for use log concentration instead concentration

        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))
        # y_data = self.Data_mask['pos_rate']
        y_data = self.Data_ana['pos_rate_average']
        return y_data.values, X, county_data

    def getX(self, init, end):
        Data_per = self.city_data[(self.city_data['Date'] >= init) & (self.city_data['Date'] <= end)]
        Data_per['NormalizedConc_s'] = Data_per['NormalizedConc_s'].fillna(method='backfill')
        x = Data_per[['NormalizedConc_s']]
        #x = np.log(Data_per[['NormalizedConc_s']])
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))
        return X

    def rate(self, beta, X):
        # x: pars, beta0, beta1, phi
        # X: covariable matriz (ww data)
        eta = X @ beta
        e_eta = np.exp(eta)
        mu = e_eta / (1 + e_eta)
        return mu

    def g(self, x):  # logistic
        mu = np.log(x / (1 - x))
        return mu

    def g_inv(self, x):  # logistic
        #e_eta = np.exp(x)
        #mu = e_eta / (1 + e_eta)
        mu= 1/(1+np.exp(-x))
        return mu

    # Out must not include the last column
    def eval_rate(self, Out, X):
        xfunc = lambda x: self.rate(x[:2], X)
        return np.apply_along_axis(xfunc, 1, Out)


    def predictive_old(self, x, X):
        nn = self.y_data.shape[0]
        #beta0, beta1,  k, init = x
        beta0, beta1,  k = x

        mu = np.zeros(nn)
        Y = np.zeros(nn)
        Xb = X[:, 1] * beta1
        #mu[0] = self.g_inv(init)
        mu[0] = self.g_inv(beta0 + Xb[0])

        a = mu[0] * self.phi
        b = self.phi - a
        Y[0] = ss.beta.rvs(a, b)

        for i in range(nn - 1):
            mu[i + 1] = self.g_inv( beta0 + Xb[i + 1] + k * self.g_inv(mu[i]))
            #mu[i + 1] = self.g_inv(beta0 + Xb[i + 1] + k * mu[i])
            a = mu[i+1] * self.phi
            b = self.phi - a
            Y[i + 1] = ss.beta.rvs(a, b)
        return Y

    def predictive(self, x, X):
        nn = self.y_data.shape[0]
        beta0, beta1,  k = x

        mu = np.zeros(nn)
        Xb = X[:, 1] * beta1

        #mu[0] = self.g_inv(beta0+ Xb[0])

        eta = np.zeros(nn)
        Y = np.zeros(nn)
        eta[0] = beta0 + Xb[0]
        mu[0] = self.g_inv(eta[0])
        a = mu[0] * self.phi
        b = self.phi - a
        Y[0] = ss.beta.rvs(a, b)

        for i in range(nn - 1):
            eta[i + 1] = beta0 + Xb[i + 1] + k * eta[i]
            #mu[i + 1] = self.g_inv(beta0 + Xb[i + 1] + k * mu[i])
            mu[i + 1] = self.g_inv(eta[i + 1])
            a = mu[i+1] * self.phi
            b = self.phi - a
            Y[i + 1] = ss.beta.rvs(a, b)
        return Y

    def loglikelihood(self, x):
        nn = self.y_data.shape[0]
        beta0, beta1,  k = x

        mu = np.zeros(nn)
        Xb = self.X[:, 1] * beta1
        eta = np.zeros(nn)
        eta[0] = beta0 + Xb[0]
        mu[0] = self.g_inv(eta[0])


        for i in range(nn- 1):
            #mu[i + 1] = self.g_inv(beta0 + Xb[i + 1] + k * self.g_inv(mu[i]) )
            #mu[i + 1] = self.g_inv(beta0 + Xb[i + 1] + k * mu[i])
            eta[i + 1] =  beta0 + Xb[i + 1] + k * eta[i] #(self.g(self.y_data[i]) - Xb[i])
            mu[i+1] = self.g_inv(eta[i+1])
        #mu = self.g_inv(eta)
        a = mu * self.phi
        b = self.phi - a
        log_likelihood = np.sum(stats.beta.logpdf(self.y_data, a, b))
        return log_likelihood



    def eval_predictive(self, Out, conc):
        xfunc = lambda x: self.predictive(x, conc)
        return np.apply_along_axis(xfunc, 1, Out)

    def mean_predictive(self, x, X):
        beta = x[:2]
        mu = self.rate(beta, X)
        return mu

    def eval_mean_predictive(self, Out, conc):
        xfunc = lambda x: self.mean_predictive(x, conc)
        return np.apply_along_axis(xfunc, 1, Out)


    def logprior(self, x):
        """
        Logarithm of a normal distribution
        """
        beta = x[:2]
        k = x[2]
        #init = x[3]
        if self.per == 0:
            cov = np.diag(self.sig)

            log_prior1 = multivariate_normal.logpdf(beta, mean=self.mu, cov=cov)
            log_prior2 = norm.logpdf(k, self.mu_k, scale=self.std_k)
            #log_prior3 = norm.logpdf(init, self.mu_init, scale=self.std_init)
        else:
            mu_b0, std_b0 = self.post_params['beta0_' + str(self.per - 1)]
            mu_b1, std_b1 = self.post_params['beta1_' + str(self.per - 1)]
            mu_k, std_k = self.post_params['k_' + str(self.per - 1)]
            #mu_init, std_init = self.post_params['init_' + str(self.per - 1)]

            log_prior1 = multivariate_normal.logpdf(beta, mean=np.array([mu_b0, mu_b1]),
                                                    cov=np.diag(np.array([std_b0 ** 2, std_b1 ** 2])))
            log_prior2 = norm.logpdf(k, mu_k, scale=std_k)
            #log_prior3 = norm.logpdf(init, mu_init, scale=std_init)

        #log_prior = log_prior1 + log_prior2 + log_prior3
        log_prior = log_prior1 +  log_prior2
        return log_prior

    def Energy(self, x):
        """
        -log of the posterior distribution
        """
        return -1 * (self.loglikelihood(x) + self.logprior(x))

    def Supp(self, x):
        """
        Support of the parameters to be estimated
        """
        return True

    def LG_Init(self):
        """
        Initial condition
        """
        if self.per == 0:
            cov = np.diag(self.sig)
            sim1 = multivariate_normal.rvs(mean=self.mu, cov=cov)
            sim2 = norm.rvs(self.mu_k, scale=self.std_k)
            #sim3 = norm.rvs(self.mu_init, scale=self.std_init)
        else:
            mu_b0, std_b0 = self.post_params['beta0_' + str(self.per - 1)]
            mu_b1, std_b1 = self.post_params['beta1_' + str(self.per - 1)]
            mu_k, std_k = self.post_params['k_' + str(self.per - 1)]
            #mu_init, std_init = self.post_params['init_' + str(self.per - 1)]

            sim1 = multivariate_normal.rvs(mean=np.array([mu_b0, mu_b1]),
                                           cov=np.diag(np.array([std_b0 ** 2, std_b1 ** 2])))
            sim2 = norm.rvs(mu_k, scale=std_k)
            #sim3 = norm.rvs(mu_init, scale=std_init)

        #sim = np.append(np.append(sim1, sim2), sim3)
        sim = np.append(sim1, sim2)

        return sim.ravel()

    def fit_posterior(self):
        scl = 1.2  # 1.2
        scl1 = 1
        #if self.county =='Yolo':
        #    scl = 1.3  # 1.2
        #    scl1 = 1.2

        beta0 = self.samples[:, 0]
        mu = beta0.mean()
        std = beta0.std() * scl
        self.post_params['beta0_' + str(self.per)] = (mu, std)

        beta1 = self.samples[:, 1]
        mu = beta1.mean()
        # std = beta1.mean() * scl
        std = beta1.std() * scl
        self.post_params['beta1_' + str(self.per)] = (mu, std)

        #phi = self.samples[:, 2]
        #mu = phi.mean()
        # std = beta1.mean() * scl
        # std = phi.std() * scl
        #std = mu * scl
        #scale = std ** 2 / mu
        #a = mu / scale
        #self.post_params['phi_' + str(self.per)] = (a, scale)

        k = self.samples[:, 2]
        mu = k.mean()
        #std = beta1.mean() * scl
        std = k.std() * scl1
        self.post_params['k_' + str(self.per)] = (mu, std)

        #init = self.samples[:, 3]
        #mu = init.mean()
        #std = beta1.mean() * scl
        #std = init.std() * scl
        #self.post_params['init_' + str(self.per)] = (mu, std)

    def RunMCMC(self, T=100000, burnin=1000):

        self.twalk = pytwalk(n=self.d, U=self.Energy, Supp=self.Supp)
        self.twalk.Run(T=T, x0=self.LG_Init(), xp0=self.LG_Init())
        self.burnin = burnin
        if self.twalk.Acc[5]!=0:
            self.iat = int(self.twalk.IAT(start=burnin)[0, 0])

            # print("\nEffective sample size: %d" % ((T-burnin)/self.iat,))
            self.samples = self.twalk.Output[burnin::(self.iat), :]  # Burn in and thining, output t-wal
            self.fit_posterior()
            print(self.post_params['beta0_' + str(self.per)])
            print(self.post_params['beta1_' + str(self.per)])
            print(self.post_params['k_' + str(self.per)])
            print("\nSaving files in ", self.savepath + self.county + '_*.pkl')
            pickle.dump(self.samples, open(self.savepath + self.county + '_per_' + str(self.per) + '_samples.pkl', 'wb'))


        else:
            print("Loading previously saved samples.")
            self.samples = pickle.load(open(self.savepath + self.county + '_per_' + str(self.per-1) + '_samples.pkl', 'rb'))
            self.fit_posterior()
            pickle.dump(self.samples, open(self.savepath + self.county + '_per_' + str(self.per) + '_samples.pkl', 'wb'))

            #?????
        outname_var = self.savepath + self.county + '_post_params.pkl'

        # print("\nSaving files in ", 'output/' + self.county + '_*.pkl')
        # pickle.dump(self.samples, open('output/' + self.county + '_per_' + str(self.per)+'_samples.pkl', 'wb'))
        # outname_var =  'output/'+ self.county + '_post_params.pkl'

        with open(outname_var, 'wb') as outfile:
            pickle.dump(self.post_params, outfile)

    def summary(self, Output_all):
        Output = Output_all[self.burnin::self.thini, :]
        Output_theta = Output[:, :self.d]
        Energy = Output[self.burnin:, -1]
        return Output_theta




