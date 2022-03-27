    
__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import fmin
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


class Nelson_Siegel_Model:
    def __init__(self):
        self.L = 0.
        self.S = 0.
        self.C = 0.
        self.l = 0.03

    def zero_yield(self, maturity, pdframe=False, table_ix=None):
        fit_rate = self.L + \
            self.S * np.divide(1 - np.exp(-maturity * self.l), maturity * self.l) + \
            self.C * (np.divide(1 - np.exp(-maturity * self.l), maturity * self.l) - np.exp(-maturity * self.l))

        if pdframe:
            rate_table = pd.DataFrame(fit_rate, columns=maturity)
            rate_table.index = table_ix
            return rate_table
        else:
            return fit_rate

    def mean_squares_error(self, params, maturity, rate):

        if not np.any(params):
            return 0.5 * np.sum((rate - self.zero_yield(maturity)) ** 2)
        else:
            self.L, self.S, self.C, self.l = params

            return 0.5 * np.sum((rate - self.zero_yield(maturity)) ** 2)


    def ols_fit(self, lambda_hat, maturity, rate):
        x1 = np.divide(1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat)
        x2 = np.divide(1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat) - np.exp(-maturity * lambda_hat)
        v_one = np.ones(x1.shape)
        X = np.c_[v_one, x1, x2]

        corr = np.linalg.inv(X.T @ X) @ X.T @ rate

        self.L, self.S, self.C = corr
        self.l = lambda_hat

        return corr

    def ols_error(self, lambda_hat, maturity, rate):
        self.ols_fit(lambda_hat, maturity, rate)
        return self.mean_squares_error(False, maturity, rate)

    def two_step_fitting(self, lambda_init, maturity, rate):
        estimated_lamnda = fmin(self.ols_error,lambda_init, args=(maturity, rate), disp=False)

        return estimated_lamnda

    def direct_fitting(self, maturity, rate):
        init = np.array([self.L,self.S,self.C,self.l])
        estimated_params = fmin(self.mean_squares_error, init, args=(maturity, rate), disp=False)

        self.L, self.S, self.C, self.l = estimated_params

        return estimated_params


class Dynamic_Nelson_Siegel_Model(Nelson_Siegel_Model):

    def ols_fit(self, lambda_hat, maturity, rate):
        x1 = np.divide(1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat)
        x2 = np.divide(1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat) - np.exp(-maturity * lambda_hat)
        v_one = np.ones(x1.shape)
        X = np.c_[v_one, x1, x2]

        corr = np.linalg.inv(X.T @ X) @ (X.T @ rate.T)
        self.L, self.S, self.C = corr.reshape(3, -1, 1)
        self.l = lambda_hat

        self.latent_factor = corr.T
        return corr

    def plot_history(self, actual_rate, fit_rate, save=False):
        maturity = actual_rate.columns
        for i in maturity:
            actual_rate[i].plot()
            fit_rate[i*12].plot()
            plt.title('The Historical Graph of Zero-Yield with {} year to maturity'.format(i))
            plt.legend(['Actual Rate', 'Fitted Nelson Siegel'])
            plt.grid()
            plt.ylabel('Yield')
            if save:
                plt.savefig('result/nelson_siegel/ns_{}year.png'.format(i), bbox_inches="tight")
            plt.show()

    def plot_latent_factor(self, save=False, name='latent_factor'):
        latent_table = pd.DataFrame(self.latent_factor, columns=['Level', 'Slope', 'Curvature'])
        latent_table.index = main_rate.index
        latent_table.plot()
        plt.grid()
        if save:
            plt.savefig('result/nelson_siegel/{}.png'.format(name), bbox_inches="tight")
        plt.show()



class Macro_Analysis:
    def __init__(self,latent_factor):
        self.latent_factor = latent_factor

    def estimate(self, macro_factor):
        x_t0 = self.latent_factor[:-1]
        x_t1 = self.latent_factor[1:]
        m_t0 = macro_factor[1:]
        v_one = np.ones([x_t0.shape[0], 1])
        X = np.c_[v_one, x_t0, m_t0]

        coef_matrix = np.linalg.inv(X.T @ X) @ (X.T @ x_t1)

        const = coef_matrix[0]
        F = coef_matrix[1:4]
        G = coef_matrix[4:]

        return const, F, G

    def macro_SSR(self, macro_factor):
        x_t0 = self.latent_factor[:-1]
        x_t1 = self.latent_factor[1:]
        m_t0 = macro_factor[1:]
        v_one = np.ones([x_t0.shape[0], 1])

        X = np.c_[v_one, x_t0, m_t0]

        coef_matrix = np.linalg.inv(X.T @ X) @ (X.T @ x_t1)

        return 0.5 * np.sum((x_t1 - X @ coef_matrix)**2,axis=0)


    def Granger_Causality(self, macro_factor, obj):
        SSR1 = self.macro_SSR(macro_factor)
        SSR0 = self.macro_SSR(np.delete(macro_factor, obj, 1))

        T = macro_factor.shape[0] - 1
        np_ = 1 + 3 + macro_factor.shape[1]
        rF = np.divide((SSR0 - SSR1), SSR1/(T - np_))

        return rF
    
    def coefficient_of_determination(self, macro_factor):
        x_t0 = self.latent_factor[:-1]
        x_t1 = self.latent_factor[1:]
        m_t0 = macro_factor[1:]
        v_one = np.ones([x_t0.shape[0], 1])

        X = np.c_[v_one, x_t0, m_t0]

        coef_matrix = np.linalg.inv(X.T @ X) @ (X.T @ x_t1)  
        
        y_hat = X @ coef_matrix
        y_bar = np.mean(x_t1,axis=0)
        
        R = 1 - np.sum((x_t1 - y_hat)**2,axis=0) / np.sum((x_t1 - y_bar)**2,axis=0)
        
        return R


if __name__ == '__main__':

    """Data Arrangement of Yields Data"""
    df_all = pd.read_csv('data/jgbcm.csv', index_col='基準日', encoding='cp932', parse_dates=True)
    df_all.index.name = 'Date'
    df_all.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40]
    df_all = df_all.replace('-', np.nan).astype(np.float32)
    df_all = df_all.resample('M').last()
    df_all = df_all[[2,5,7,10,15,20]]
    main_rate = df_all.dropna(axis=0)

    """Data Arrangement of Macro Data"""
    macro_var = ['CPI','NK','CGPI','MBA','UP','CI','US10','FF']
    macro_data = pd.read_csv('data/macro_data2.csv', index_col=0, encoding='cp932', parse_dates=True)
    macro_data = macro_data.replace('ND', np.nan).astype(np.float32)
    #macro_data = macro_data.dropna()
    macro_data = macro_data.resample('M').last()
    macro_data = macro_data.diff()
    macro_data.columns = macro_var
    macro_data.index.name = 'Date'
    macro_data = macro_data.dropna()

    """Create the Available Data"""
    full_period = macro_data.index & main_rate.index
    macro_data = macro_data.reindex(full_period)
    main_rate = main_rate.reindex(full_period)
    maturity = 12 * main_rate.columns.values
    rate = main_rate.values

    """Modeling with Dynamic Nelson Siegel"""
    dns = Dynamic_Nelson_Siegel_Model()
    estimated_lambda = dns.two_step_fitting(0.008, maturity, rate)
    latent_factor = dns.ols_fit(estimated_lambda, maturity, rate)
    fit_rate = dns.zero_yield(maturity, True, main_rate.index)
    dns.plot_history(main_rate, fit_rate, save=True)
    dns.plot_latent_factor(save=False)

    """Analysis of Macro Factor"""
    ma = Macro_Analysis(latent_factor.T)
    const, F, G = ma.estimate(macro_data.values)
    F_stats = ma.Granger_Causality(macro_data.values, 1)
    granger_test = F_stats > stats.distributions.chi2.ppf(0.95, 2)

    """Error"""
    err = ((main_rate[10].values - fit_rate[12 * 10].values) ** 2 ) / main_rate.shape[1]
    err = pd.DataFrame(err)
    err.index = main_rate.index
    plt.plot(err)
    plt.xlabel('Date')
    plt.ylabel('MSE')
    plt.title('Mean Squares Error on the yield to maturity 10 year')
    plt.grid()
    plt.savefig('result/nelson_siegel/err_10y.png', bbox_inches="tight")
    plt.show()

    """Fitting test"""
    test_m = np.arange(1, 21)
    fit_rate = dns.zero_yield(test_m*12,True,main_rate.index)
    test_r = 318

    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter ('%.1f'))
    plt.scatter(main_rate.columns.values, main_rate.iloc[test_r].values, label='Original points')
    plt.plot(test_m, fit_rate.iloc[test_r].values,'r', label='Nelson Siegel Fitting')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Maturity')
    plt.ylabel('Rate')
    plt.xticks(np.arange(1,11)*2)
    plt.title('Fitting on {}_{}'.format(fit_rate.iloc[test_r].name.month,fit_rate.iloc[test_r].name.year))
    plt.savefig('result/nelson_{}_{}.png'.format(fit_rate.iloc[test_r].name.month,fit_rate.iloc[test_r].name.year), bbox_inches="tight")
    plt.show()


