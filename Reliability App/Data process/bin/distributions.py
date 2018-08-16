import numpy as np
import pandas as pd
import warnings
import qgrid
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
import plotly.offline as py

py.init_notebook_mode(connected=True)

warnings.filterwarnings('ignore')


# ----------------------------------- #
#            Laplace Test             #
# ----------------------------------- #


def test_laplace(failure_times, alpha=0.05):
    """
    For a given vector of failure times, returns the Laplace Test result
    , resulting in a string that returns if times are IID or not.

    :param failure_times: Failure times vector (np.array)
    :param alpha: Confidence level (float)
    :return: Trend of the data (string)
    """

    if len(failure_times) < 4:

        raise ValueError("Sample size is not big enough for Laplace Test for "
                         "trend (minimum size = 4).")

    else:

        total_time = failure_times.sum()
        number_of_times = len(failure_times)
        sum_elapsed_times = failure_times.cumsum().sum() - total_time

        test_score = np.sqrt(12 * (number_of_times-1)) * \
                            (sum_elapsed_times / ((number_of_times-1)
                                                  * total_time) - 0.5)

        back_limit, front_limit = stats.norm.interval(1 - alpha, 0, 1)[0], \
            stats.norm.interval(1 - alpha, 0, 1)[1]

        if test_score <= back_limit:
            return 'Positive Trend'
        elif test_score >= front_limit:
            return 'Negative Trend'
        else:
            return 'IID Times'


# ----------------------------------- #
#              PlotProb               #
# ----------------------------------- #
def prob_plot(dist_fun, params, t, CI=0.95,
              plot_title='', ax=None):
    r"""
    Probability plot. Similar to QQ plot.

    :param dist_fun: an object of the initialized distribution function.
    :param params: distribution law parameter. eg. shape, location, etc.
    :param t: a vector of the time values.
    :param CI: a float belonging to the interval of ]0,1], representing the
     Confidence Interval for the fitted line.
    :param plot_title: a string with the title to be displayed in the plot.
     Optional.
    :param ax: an object containing an axes of matplotlib. Optional.
    :return ax: an object containing an axes of matplotlib and the drawn plot.
    """
    i = np.arange(1, len(t) + 1)
    t_quantiles = np.asarray((i - 0.3) / (len(i) + 0.4))
    y_qq = np.sort(t)
    x_qq = dist_fun.ppf(params, t_quantiles)

    # fit line
    z = np.polyfit(x_qq, y_qq, 1)

    p_y = x_qq * z[0] + z[1]

    # get the coordinates for the fit curve
    c_y = [np.min(p_y), np.max(p_y)]
    c_x = [np.min(x_qq), np.max(x_qq)]

    # calculate the y-error (residuals)
    y_err = y_qq - p_y

    # create series of new test x-values to predict for

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x_qq)  # mean of x
    n = len(x_qq)  # number of samples in original fit
    t = stats.t.ppf(1 - ((1 - CI) / 2), n)
    s_err = np.sum(y_err ** 2)  # sum of the squares of the residual

    confs = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + ((x_qq - mean_x) ** 2) /
                                             ((np.sum(x_qq ** 2)) - n * (mean_x ** 2))))

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    # set-up the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    # else, use the argument

    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Ordered Values')

    # plot sample data
    ax.plot(x_qq, y_qq, '+', label='Sample observations')

    # plot line of best fit
    ax.plot(c_x, c_y, 'r-',
            # label='Regression line'
            )

    # plot confidence limits
    ax.plot(x_qq, lower, 'b--',
            label='Lower confidence limit ({}%)'.format(int(CI * 100))
            )
    ax.plot(x_qq, upper, 'b--',
            label='Upper confidence limit ({}%)'.format(int(CI * 100))
            )

    ax.set_title('Probability Plot ' + plot_title)

    return ax


# ----------------------------------- #
#                ECDF                 #
# ----------------------------------- #
def ecdf(x):
    r"""
    Empirical Cumulative Distribution Function for fitting a distribution. It is
    required for fitting a distribution using the least squares method and for
    the Kolmogorov-Smirnov test.

    :param x: a list, or an array of x values for computing the probabilities.
    :return: x_unique, cdf_prob
    """
    x_unique, counts = np.unique(x, return_counts=True)
    ecdf_prob = counts.cumsum() / len(x)

    return x_unique, ecdf_prob


# ----------------------------------- #
#       Kolmogorov-Smirnov test       #
# ----------------------------------- #
def ks_test(dist_fun, params, x):
    r"""
    Compute Kolmogorov-Smirnov test. The null hypothesis is that 2 independent
    samples are drawn from the same continuous distribution. Therefore, this
    function returns the p-value of this statistical test. The null hypothesis
    is that 2 independent samples are drawn from the same continuous
    distribution.

    :param dist_fun: the initialized class of the distribution function.
    eg. analytics_stat.distributions.Weibull()
    :param params: a list of parameters defining the distribution function.
    If the length of parameter is lower than 3, than the parameters should be
    placed in the given order.
    :param x: a number, a list, or an array of x values for computing the
    probabilities.

    :return: p-value regarding the hypothesis testing.
    """

    # calculate the empirical cdf
    x_unique, ecdf_prob = ecdf(x)

    # calculate the cdf of the fitted params
    cdf_prob = dist_fun.cdf(params, x_unique)

    # compute Kolmogorov-Smirnov test
    # The null hypothesis is that 2 independent samples are drawn from the same
    # continuous distribution.
    _, p_value = stats.ks_2samp(ecdf_prob, cdf_prob)

    return p_value


# ----------------------------------- #
#                R^2                  #
# ----------------------------------- #
def r_sq(dist_fun, params, x):
    """
    R-squared, or coefficient of determination is a metric for goodness-of-fit which returns the proportion of the
    variance that is explained by the linear model of target vs. prediction.
    :param dist_fun: an object of the initialized distribution function.
    :param params: a list of parameters defining the distribution function. If the length of parameter is lower
    than 3, than the parameters should be placed in the given order.
    :param x: a number, a list, or an array of x values for computing the probabilities.
    :return:
    """
    # calculate the empirical cdf
    x_unique, ecdf_prob = ecdf(x)

    # calculate the cdf of the fitted params
    cdf_prob = dist_fun.cdf(params, x_unique)

    # compute R^2
    return np.corrcoef(cdf_prob, ecdf_prob)[0, 1] ** 2


# ----------------------------------- #
#         rank distributions          #
# ----------------------------------- #
def rank_dist(x, x_censored=None, method='MLE', distributions=None, display=True):
    r"""
        Rank distributions according to the goodness-of-fit.
        Available distributions:
            1.	Weibull 2 parameters        Weibull(loc=0)
            2.	Weibull 3 parameters        Weibull()
            3.	Normal                      Normal()
            4.	Log-Normal                  Lognormal()
            5.	Exponential 1 parameter     Exponential(loc=0)
            6.	Exponential 2 parameter     Exponential()
            7.	Logistic                    Logistic()
            8.	Log-logistic                Loglogistic()


    :param x: a 1-D array with the data to be fitted.
    :param x_censored: a 1-D array with the right-censored data to be fitted.
    :param method: Options: 'MLE' or 'LSQ', meaning, respectively, the Maximum Likelihood Estimation and the Least Squares
    methods for fitting distribution laws to the data.
    :param distributions: a list containing two or more of the available distributions: 'weibull3p', 'weibull2p',
                        'normal', 'lognormal', 'exponential1p', 'exponential2p', 'logistic', 'loglogistic'.
                        Default: None, i.e. all
    :param display: Display data frame with results. Default: True.
    :return: results: a dictionary with the results of the goodness of fit.
    """

    result = dict()

    dist_names = ['weibull3p', 'weibull2p', 'normal', 'lognormal',
                  'exponential1p', 'exponential2p', 'logistic', 'loglogistic']

    dist_funs = [Weibull(),
                 Weibull(loc=0),
                 Normal(),
                 Lognormal(),
                 Exponential(loc=0),
                 Exponential(),
                 Logistic(),
                 Loglogistic()]

    if distributions is None:
        # use all distributions available

        dist_index = range(len(dist_funs))

    elif (type(distributions) == list) & (len(distributions) >= 2):

        dist_index = [i for i, n in enumerate(dist_names) if n in distributions]

    else:
        raise ValueError("Incorrect specification of distributions.")

    if method is None:
        # use all methods available
        ''' 1. Maximum Likelihood Estimation    'mle'
            2. Least Squares                    'lsq'
        '''
        pass

    elif method.lower() == 'mle':  # Maximum Likelihood Estimation
        for dist_i in dist_index:
            result[dist_names[dist_i]] = dict()
            result[dist_names[dist_i]]['params'] = dist_funs[dist_i].fit(x, x_censored=x_censored, method='mle')

            result[dist_names[dist_i]]['goodness_of_fit'] = dict()

            result[dist_names[dist_i]]['goodness_of_fit']['lk'] = dist_funs[dist_i].log_lk(
                result[dist_names[dist_i]]['params'], x)

            result[dist_names[dist_i]]['goodness_of_fit']['ks'] = ks_test(
                dist_funs[dist_i], result[dist_names[dist_i]]['params'], x)

            result[dist_names[dist_i]]['goodness_of_fit']['r_sq'] = r_sq(
                dist_funs[dist_i], result[dist_names[dist_i]]['params'], x)

        if display is True:
            res_df = pd.DataFrame(index=result.keys())

            res_df['LK'] = [result[k]['goodness_of_fit']['lk'] for k in result.keys()]
            res_df['R_sq'] = [result[k]['goodness_of_fit']['r_sq'] for k in result.keys()]
            res_df['KS'] = [result[k]['goodness_of_fit']['ks'] for k in result.keys()]

            res_df['rank dist MLE'] = (
                    res_df['LK'].rank(ascending=False) * .5 + res_df['R_sq'].rank(ascending=False) * .1 + res_df[
                'KS'].rank(
                ascending=False) * .4).rank()

            qgrid.show_grid(res_df.sort_values('rank dist MLE'))

    elif method.lower() == 'lsq':  # Least Squares

        for dist_i in dist_index:
            # print(dist_funs)
            result[dist_names[dist_i]] = dict()
            result[dist_names[dist_i]]['params'] = dist_funs[dist_i].fit(x, method='lsq')

            result[dist_names[dist_i]]['goodness_of_fit'] = dict()

            result[dist_names[dist_i]]['goodness_of_fit']['lk'] = dist_funs[dist_i].log_lk(
                result[dist_names[dist_i]]['params'], x)

            result[dist_names[dist_i]]['goodness_of_fit']['ks'] = ks_test(
                dist_funs[dist_i], result[dist_names[dist_i]]['params'], x)

            result[dist_names[dist_i]]['goodness_of_fit']['r_sq'] = r_sq(
                dist_funs[dist_i], result[dist_names[dist_i]]['params'], x)

        if display is True:
            res_df = pd.DataFrame(index=result.keys())

            res_df['LK'] = [result[k]['goodness_of_fit']['lk'] for k in result.keys()]
            res_df['R_sq'] = [result[k]['goodness_of_fit']['r_sq'] for k in result.keys()]
            res_df['KS'] = [result[k]['goodness_of_fit']['ks'] for k in result.keys()]

            res_df['rank dist LSQ'] = (
                    res_df['LK'].rank(ascending=False) * .3 + res_df['R_sq'].rank(ascending=False) * .2 + res_df[
                'KS'].rank(
                ascending=False) * .5).rank()

            qgrid.show_grid(res_df.sort_values('rank dist LSQ'))

    else:
        raise ValueError('The method specified is not defined.')

    return result  # a dictionary or a data frame with rankings for each method


# ----------------------------------- #
#          combine rankings           #
# ----------------------------------- #
def combine_ranks(res_mle, res_ls, display=True):
    """
    A function for combining the rankings provided by both fitting methods, i.e. MLE and LSQ.

    :param res_mle: a dictionary of the fitting result via Maximum Likelihood Estimation.
    :param res_ls: a dictionary of the fitting result via Least Squares.
    :param display: Display data frame with results. Default: True.
    :return: three pandas DataFrames of both fitting methods ranking and the overall ranking in the order of All,
     MLE, LSQ.
    """

    # data frame of MLE results:
    res_df_mle = pd.DataFrame(index=res_mle.keys())
    res_df_mle['LK'] = [res_mle[k]['goodness_of_fit']['lk'] for k in res_mle.keys()]
    res_df_mle['R_sq'] = [res_mle[k]['goodness_of_fit']['r_sq'] for k in res_mle.keys()]
    res_df_mle['KS'] = [res_mle[k]['goodness_of_fit']['ks'] for k in res_mle.keys()]

    # data frame of least squares results:
    res_df_ls = pd.DataFrame(index=res_ls.keys())
    res_df_ls['LK'] = [res_ls[k]['goodness_of_fit']['lk'] for k in res_ls.keys()]
    res_df_ls['R_sq'] = [res_ls[k]['goodness_of_fit']['r_sq'] for k in res_ls.keys()]
    res_df_ls['KS'] = [res_ls[k]['goodness_of_fit']['ks'] for k in res_ls.keys()]

    ranking_all = pd.DataFrame()

    ranking_all['rank dist ALL'] = (
                res_df_mle['LK'].rank(ascending=False) * .5 + res_df_ls['LK'].rank(ascending=False) * .3 +
                res_df_mle['R_sq'].rank(ascending=False) * .1 + res_df_ls['R_sq'].rank(ascending=False) * .2 +
                res_df_mle['KS'].rank(ascending=False) * .4 + res_df_ls['KS'].rank(ascending=False) * .5).rank()

    if display is True:
        qgrid.set_grid_option('fullWidthRows', True)
        qgrid.show_grid(ranking_all.sort_values('rank dist ALL'))

    return ranking_all, res_df_mle, res_df_ls


# ----------------------------------- #
#           plot best dist            #
# ----------------------------------- #
def plot_best(ranking_all, x):
    """
    A function for plotting the distribution law summary (PDF, 1-CDF, and PPF) for both fitting methods.

    :param ranking_all: A pandas DataFrame of the overall ranking result.
    :param x: a 1-D array with the data to be fitted.
    :return: None.
    """

    dist_names = ['weibull3p', 'weibull2p', 'normal', 'lognormal',
                  'exponential1p', 'exponential2p', 'logistic', 'loglogistic']

    best_dist = ranking_all.sort_values('rank dist ALL').sort_values('rank dist ALL').index[0]

    dist_index = [i for i, n in enumerate(dist_names) if n == best_dist]

    dist_funs = [Weibull(),
                 Weibull(loc=0),
                 Normal(),
                 Lognormal(),
                 Exponential(loc=0),
                 Exponential(),
                 Logistic(),
                 Loglogistic()]

    params_dist_mle = dist_funs[dist_index[0]].fit(x, method='MLE')
    params_dist_lsq = dist_funs[dist_index[0]].fit(x, method='lsq')

    if dist_names[dist_index[0]] in ['normal', 'logistic']:
        x_plot = np.linspace(x.min() - x.min() * .5, x.max() + x.max() * .5, 100)
    else:
        x_plot = np.linspace(1e-6, x.max() + x.max() * .5, 100)

    pdf_mle = dist_funs[dist_index[0]].pdf(params_dist_mle, x_plot)

    pdf_lsq = dist_funs[dist_index[0]].pdf(params_dist_lsq, x_plot)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].plot(x_plot, pdf_mle, '--')
    ax[0, 0].plot(x_plot, pdf_lsq, '--')

    ax[0, 0].hist(x, normed=True, alpha=.5)
    ax[0, 0].legend(['MLE', 'LSQ', 'Histogram data'])

    # plt.show()
    ax[0, 0].set_title('Best Fit: {},\n PDF'.format(best_dist))

    print('MLE params: {}'.format(params_dist_mle))
    print('LSQ params: {}'.format(params_dist_lsq))

    ax[0, 1].plot(x_plot, 1 - dist_funs[dist_index[0]].cdf(params_dist_mle, x_plot))
    ax[0, 1].plot(x_plot, 1 - dist_funs[dist_index[0]].cdf(params_dist_lsq, x_plot))

    ax[0, 1].set_yticks(np.arange(0, 1.05, .05))
    # plt.xticks(range(0, int(np.max(x_plot)), 5))

    ax[0, 1].grid()
    ax[0, 1].legend(['MLE', 'LSQ'])
    ax[0, 1].set_title('Reliability Function 1-CDF')
    ax[0, 1].set_xlabel('time (t)')
    ax[0, 1].set_ylabel('Reliability')

    # Probplot (similar to QQ)
    probplot(dist_funs[dist_index[0]], params_dist_lsq, x, plot_title='LSQ', ax=ax[1, 1])
    probplot(dist_funs[dist_index[0]], params_dist_mle, x, plot_title='MLE', ax=ax[1, 0])

    try:
        plt.tight_layout()
    except:
        print('cannot use tight layout.')
    plt.show()

    return


# ----------------------------------- #
#              WEIBULL                #
# ----------------------------------- #
class Weibull(object):
    def __init__(self, shape=None, scale=None, loc=None):
        r"""
        Initialize the weibull distribution function. You can fix any of the 3
        parameters by passing it as the argument.
        This might be especially useful for MLE of parameter for specific
         parameters at a time.

        :param shape: Also known as Beta parameter.
        :param scale: Also known as Mu parameter.
        :param loc: Also known as Gamma parameter.
        """
        self._params = np.array([shape, scale, loc])

        self._not_params_i = [i for i, f in enumerate(self._params)
                              if f is None]  # index of not defined params

    def fix(self, shape=None, scale=None, loc=None):
        self._params = np.array([shape, scale, loc])

        self._not_params_i = [i for i, f in enumerate(self._params)
                              if f is None]  # index of not defined params

    # ----------------------------------- #
    #                  PDF                #
    # ----------------------------------- #
    def pdf(self, params, x):
        r"""
        Computes the PDF of Weibull 2 or 3 parameters.

        :param params: a list of parameters containing shape, scale and
        location. If the length of parameter is lower
        than 3, than the parameter should be placed in order.
        :param x: a number, a list, or an array of x values for computing
        probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params

        else:
            print('')
            print('Len params: {}'.format(len(params)))
            print(params)
            print('Len self._params: {}'.format(len(self._params)))
            print(self._params)
            print('not in params: {}'.format(self._not_params_i))
            raise ValueError('The size of parameters is not correct.')

        x = np.array(x)

        shape = self._params[0]
        scale = self._params[1]
        loc = self._params[2]

        prob = np.nan_to_num((shape / scale) * ((x-loc) / scale) **
                             (shape-1) * np.exp(-((x-loc) / scale) ** shape))

        return prob

    # ----------------------------------- #
    #                 CDF                 #
    # ----------------------------------- #
    def cdf(self, params, x):
        r"""
        Computes the CDF of Weibull 2 or 3 parameters.

        :param params: a list of parameters containing 1) shape, 2) scale and 3)
        location. If the length of parameter is lower
        than 3, than the parameters should be placed in the given order.
        :param x: a number, a list, or an array of x values for computing
        probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))

            raise ValueError('The size of parameters is not correct.')

        x = np.array(x)

        shape = self._params[0]
        scale = self._params[1]
        loc = self._params[2]

        return np.nan_to_num(1 - np.exp(-((x - loc) / scale) ** shape))

    # ----------------------------------- #
    #                1-CDF                #
    # ----------------------------------- #
    def inv_cdf(self, params, x):
        r"""
        Computes the inverse of CDF of Weibull 2 or 3 parameters.

        :param params: a list of parameters containing 1) shape, 2) scale and 3)
        location. If the length of parameter is lower than 3, than the
         parameters should be placed in the given order.
        :param x: a number, a list, or an array of x values for
        computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))

            raise ValueError('The size of parameters is not correct.')

        shape = self._params[0]
        scale = self._params[1]
        loc = self._params[2]

        return np.nan_to_num(np.exp(-((x - loc) / scale) ** shape))

    # ----------------------------------- #
    #                 PPF                 #
    # ----------------------------------- #
    def ppf(self, params, q):
        r"""
        Computes the PPF.

        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))

            raise ValueError('The size of parameters is not correct.')

        shape = self._params[0]
        scale = self._params[1]
        loc = self._params[2]

        return stats.weibull_min.ppf(q, shape, loc, scale)

    # ----------------------------------- #
    #              neg_log_lk             #
    # ----------------------------------- #
    def neg_log_lk(self, params, x, x_censored=None):
        r"""
        Computes negative sum of log-likelihood for the probability of x given
        the distribution parameters of Weibull 2 or 3 parameters. This function
        is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is
        required to convert product of the Likelihood function in a sum.

        :param params: a list of parameters containing shape, scale and
        location. If the length of parameter is lower than 3, then the parameter
        should be placed in order.
        :param x: a number, a list, or an array of x values for computing
        probability.
        :param x_censored: a number, a list, or an array of x values for
        computing cumulative probability.
        :return: negative sum of log-likelihood for the probability of x given
        the distribution parameters.
        """
        # Uncensored probability
        prob_x = np.sum(np.log(np.nan_to_num(self.pdf(params, x))))
        # Censored probability
        if x_censored:
            prob_x_censored = np.sum(np.log(np.nan_to_num(self.cdf(params,
                                                                   x_censored)))
                                     )
        else:
            prob_x_censored = 0

        return -np.nansum(prob_x) - np.nansum(prob_x_censored)

    # ----------------------------------- #
    #                log_lk               #
    # ----------------------------------- #
    def log_lk(self, params, x, x_censored=None):
        r"""
        Computes sum of log-likelihood for the probability of x given the
        distribution parameters of Weibull 2 or 3 parameters. This function is
        useful for model validation, i.e. goodness-of-fit. The log
        transformation is required to convert product of the Likelihood
        function in a sum.

        :param params: a list of parameters containing shape, scale and
        location. If the length of parameter is lower  than 3, than the
        parameter should be placed in order.
        :param x: a number, a list, or an array of x values for computing
        probability.
        :param x_censored: a number, a list, or an array of x values for
        computing cumulative probability.
        :return: negative sum of log-likelihood for the probability of x given
        the distribution parameters.
        """
        # Uncensored probability
        prob_x = np.sum(np.nan_to_num(self.pdf(params, x)))
        # Censored probability
        prob_x_censored = np.sum(np.nan_to_num(self.cdf(params, x_censored)))

        return np.nansum(np.log(prob_x + prob_x_censored))

    # ----------------------------------- #
    #                 fit                 #
    # ----------------------------------- #
    def fit(self, x, x_censored=None, method='MLE', implementation='nano'):
        r"""
        Fits the distribution to the data returning the parameters defining the
        Weibull distribution. By default it estimates the 3 parameters via MLE.
        The user may choose the regression method for fitting, as well as fixing
        any of the 3 parameters for finding the remainder.

        :param x: a 1D array of data to be fitted.
        :param x_censored: a 1D array of censored data to be fitted.
        :param method: Maximum Likelihood Estimation, or Least Squares
        Regression.
        :param implementation: (for 3 parameter fitting only) Depending on the
        implementation results might vary slightly, although being
        mathematically acceptable.
        :return: a list of the fitted parameters [shape, scale, loc].
        """

        params = list()

        # If the initialization was made with no fixed parameters
        if len(self._not_params_i) == 3:
            # TODO Clean the code at this part (a lot of ifs and checks)
            # Most likelihood estimation method
            if method.lower() == 'mle':

                if implementation.lower() == 'reliasoft':
                    # First fix location at 0 and find shape and scale.
                    self.fix(loc=0)

                    res_1 = opt.minimize(self.neg_log_lk, [1, np.mean(x)], args=(x,),
                                         # method='L-BFGS-B',
                                         method='Nelder-Mead'
                                         )

                    # print(res_1.success)
                    params_init = res_1.x
                    # print('Params 1): {}'.format(params_init))

                    # Second fix shape and scale and find Location
                    self.fix(shape=params_init[0], scale=params_init[1])

                    res_2 = opt.minimize(self.neg_log_lk, [np.min(x)], args=(x,),
                                         # method='L-BFGS-B'
                                         method='Nelder-Mead'
                                         )

                    # print(res_2.success)
                    loc_rs = res_2.x[0]
                    print('loc 2): {}'.format(res_2.x))

                    # third locally optimize shape and scale  by gradient method, fixing location
                    self.fix(loc=loc_rs)

                    res_3 = opt.minimize(self.neg_log_lk, params_init, args=(x,))

                    print('params: {}'.format(res_3.x))

                    # print(res_3.success)
                    params = list(np.append(res_3.x, loc_rs))

                elif implementation.lower() == 'nano':

                    """ Bound assumptions:
                    For shape (beta), a value between a small number and two is 
                    what is wanted to fit, since in that case, the failure rate
                    is in limit slightly increasing over time.
                    
                    For scale (mu), the value must be between one (smallest 
                    measurement possible, assuming there are no decimals) and
                     the maximum of values of our failure event array. 
                    
                    Location (gamma) has no meaning for now(...) 
                    """
                    # TODO Get better assumptions for beta and gamma (ours)
                    bounds_weibull = [(1e-6, 2), (1, np.max(x)), (0, 3 - 1e-6)]

                    # first globally optimize all parameter by differential
                    # evolution to avoid local minimum
                    self.fix()  # initialize params
                    res_init = opt.differential_evolution(self.neg_log_lk,
                                                          bounds=bounds_weibull,
                                                          args=(x, x_censored,))
                    print(res_init.x)
                    # second locally optimize all parameter by gradient method
                    # to converge to the local minimum
                    res_opt = opt.minimize(self.neg_log_lk, res_init.x,
                                           args=(x, x_censored),
                                           # method='L-BFGS-B',
                                           # method='Nelder-Mead',
                                           bounds=bounds_weibull
                                           )
                    print(res_opt.x)
                    params_global = list(res_opt.x)

                    # model 2) Reliasoft: ----------------
                    # First fix location at 0 and find shape and scale
                    self.fix(loc=0)

                    res_1 = opt.minimize(self.neg_log_lk, [1, np.mean(x)],
                                         args=(x,x_censored,),
                                         # method='L-BFGS-B',
                                         method='Nelder-Mead'
                                         )
                    print(res_1.x)
                    # print(res_1.success)
                    params_init = res_1.x
                    # print('Params 1): {}'.format(params_init))

                    # Second fix shape and scale and find Location
                    self.fix(shape=params_init[0], scale=params_init[1])

                    res_2 = opt.minimize(self.neg_log_lk, [np.min(x)], args=(x,x_censored,),
                                         # method='L-BFGS-B'
                                         method='Nelder-Mead'
                                         )

                    # print(res_2.success)
                    loc_rs = res_2.x[0]
                    # print('loc 2): {}'.format(loc_our))

                    # third locally optimize shape and scale  by gradient
                    # method, fixing location
                    self.fix(loc=loc_rs)

                    res_3 = opt.minimize(self.neg_log_lk, params_init,
                                         args=(x,x_censored,))

                    # print(res_3.success)
                    params_rs = np.append(res_3.x, loc_rs)
                    print(params_rs)
                    p_value_global = ks_test(Weibull(), params_global, x)
                    print(p_value_global)
                    # make ensemble
                    params = list(
                        np.sum([(1-p_value_global) * np.array(params_rs), p_value_global * np.array(params_global)], axis=0))  # weighted sum

                    self.fix()

                elif implementation.lower() == 'global':
                    # My method for MLE of parameters in a 3 parameter weibull.

                    # TODO talk about these bounds assumptions
                    bounds_weibull = [(1e-6, 2), (1, np.max(x)), (0, 3 - 1e-6)]

                    # first globally optimize all parameter by differential evolution to avoid local minima
                    self.fix()  # initialize params
                    res_init = opt.differential_evolution(self.neg_log_lk,
                                                          # popsize=10,
                                                          bounds=bounds_weibull, args=(x,))

                    # second locally optimize all parameter by gradient method to converge to the local minimum
                    res_opt = opt.minimize(self.neg_log_lk, res_init.x, args=(x,),
                                           # method='L-BFGS-B',
                                           # method='Nelder-Mead',
                                           # bounds=bounds_weibull
                                           )

                    params = list(res_opt.x)

                else:
                    raise ValueError('Implementation selected is not defined for 3 parameter Weibull.')

            elif method.lower() == 'lsq':

                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # 2) create error function:
                # errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function p: params, t: x
                #
                # 3) find parameters:
                # params, _ = opt.leastsq(errfunc, [1, np.mean(x), np.min(x)], args=(x_unique, ecdf_prob))
                # params = list(params)
                bounds_weibull = [(1e-6, 2), (1, np.max(x)), (0, 3 - 1e-6)]

                res_init = opt.differential_evolution(self.neg_log_lk,
                                                      # popsize=10,
                                                      bounds=bounds_weibull, args=(x,))

                errfunc = lambda p, t, y: np.sum((self.cdf(p, t) - y) ** 2)  # sse

                res_1 = opt.minimize(errfunc, res_init.x, args=(x_unique, ecdf_prob),
                                     method='L-BFGS-B',
                                     # method='Nelder-Mead',
                                     bounds=bounds_weibull
                                     )

                params = list(res_1.x)

            self.fix()

        elif (len(self._not_params_i) <= 2) & (len(self._not_params_i) != 0):
            init_params = np.array([0, np.mean(x), np.min(x)])[self._not_params_i]
            # print('not_params:', self._not_params_i)
            # print('init_params:', init_params)
            bounds_weibull = np.array([(1e-6, 2), (1, np.max(x)), (0, 3 - 1e-6)])[self._not_params_i]
            # ('bounds_weibull:', bounds_weibull)
            if implementation is None:
                implementation = 'nano'

            if method.lower() == 'mle':
                if implementation.lower() == 'nano':
                    res_1 = opt.minimize(self.neg_log_lk, init_params, args=(x,x_censored),
                                         #method='L-BFGS-B',
                                         method='Nelder-Mead',
                                         bounds=bounds_weibull
                                         )

                    # print(res_1.success)
                    params = list(res_1.x)

                if implementation.lower() == 'scipy':
                    # [self._not_params_i]  # index of parameters to find. [shape:0, scale:1, loc:2]
                    # self._params  # params defined [shape, scale, loc]

                    if len(self._not_params_i) == 1:

                        if 0 in self._not_params_i:
                            shape, _, _ = stats.weibull_min.fit(x, fscale=self._params[1], floc=self._params[2])
                            params = [shape]

                        elif 1 in self._not_params_i:
                            _, _, scale = stats.weibull_min.fit(x, f0=self._params[0], floc=self._params[2])
                            params = [scale]

                        elif 2 in self._not_params_i:
                            _, loc, _ = stats.weibull_min.fit(x, f0=self._params[0], fscale=self._params[1])
                            params = [loc]

                    elif len(self._not_params_i) == 2:

                        if (0 in self._not_params_i) & (1 in self._not_params_i):
                            shape, _, scale = stats.weibull_min.fit(x, floc=self._params[2])
                            params = [shape, scale]

                        elif (0 in self._not_params_i) & (2 in self._not_params_i):
                            shape, loc, _ = stats.weibull_min.fit(x, fscale=self._params[1])
                            params = [shape, loc]

                        elif (1 in self._not_params_i) & (2 in self._not_params_i):
                            _, loc, scale = stats.weibull_min.fit(x, f0=self._params[0])
                            params = [scale, loc]

            elif method.lower() == 'lsq':  # TODO test this
                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # # 2) create error function:
                # errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function
                #
                # # 3) find parameters:
                # params, _ = opt.leastsq(errfunc, init_params, args=(x_unique, ecdf_prob))
                # params = list(params)

                res_init = opt.differential_evolution(self.neg_log_lk,
                                                      # popsize=10,
                                                      bounds=bounds_weibull, args=(x,))

                errfunc = lambda p, t, y: np.sum((self.cdf(p, t) - y) ** 2)  # sse

                res_1 = opt.minimize(errfunc, res_init.x, args=(x_unique, ecdf_prob),
                                     method='L-BFGS-B',
                                     # method='Nelder-Mead',
                                     bounds=bounds_weibull
                                     )

                params = list(res_1.x)


            else:
                raise ValueError('Method selected is not defined.')

            self.fix(loc=0)

        else:
            raise ValueError('Unacceptable number of parameters.')

        return params


# ----------------------------------- #
#               Normal                #
# ----------------------------------- #
class Normal(object):
    def __init__(self, mu=None, sigma=None):
        r"""
        Initialization and fixing the paramenter of a Normal (Gaussian) distribution. It is usually described by two
        parameters: mean, ($\mu$), and standard deviation, ($\sigma$). If no value is fixed, the computation is direct
        without requiring optimization. Otherwise, the remaining parameter is estimated via Maximum Likelihood Estimation.

        :param mu: Mean.
        :param sigma: Standard Deviation
        """
        self._params = np.array([mu, sigma])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

    # ----------------------------------- #
    #                fix                  #
    # ----------------------------------- #
    def fix(self, mu=None, sigma=None):
        self._params = np.array([mu, sigma])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

    # ----------------------------------- #
    #                 PDF                 #
    # ----------------------------------- #
    def pdf(self, params, x):
        r"""
        Computes the PDF of Normal distribution. A Normal (Gaussian) distribution is usually described by two
        parameters: mean, ($\mu$), and standard deviation, ($\sigma$).

        $f(x;\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        mu = self._params[0]
        std = self._params[1]

        return stats.norm.pdf(x, mu, std)

    # ----------------------------------- #
    #                 CDF                 #
    # ----------------------------------- #
    def cdf(self, params, x):
        r"""
        Computes the CDF of a Normal distribution.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        mu = self._params[0]
        std = self._params[1]

        return stats.norm.cdf(x, mu, std)

    # ----------------------------------- #
    #                 PPF                 #
    # ----------------------------------- #
    def ppf(self, params, q):
        r"""
        Computes the PPF.

        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        mu = self._params[0]
        std = self._params[1]

        return stats.norm.ppf(q, mu, std)

    # ----------------------------------- #
    #              neg_log_lk             #
    # ----------------------------------- #
    def neg_log_lk(self, params, x):
        r"""
        Computes negative sum of log-likelihood for the probability of x given the distribution parameters. This
        function is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is required to convert produtory of
        the Likelihood function in a sum.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        mu = self._params[0]
        std = self._params[1]

        return -np.nansum(stats.norm.logpdf(x, mu, std))

    # ----------------------------------- #
    #               log_lk                #
    # ----------------------------------- #
    def log_lk(self, params, x):
        r"""
        Computes sum of log-likelihood for the probability of x given the distribution parameters.
         This function is useful for model validation, i.e. goodness-of-fit. The log transformation is
        required to convert produtory of the Likelihood function in a sum.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        mu = self._params[0]
        std = self._params[1]

        return np.nansum(stats.norm.logpdf(x, mu, std))

    # ----------------------------------- #
    #                 fit                 #
    # ----------------------------------- #
    def fit(self, x,
            method='mle'):  # TODO add method and implementation kwds (if really necessary, or for homogeneity of functions)
        r"""
        Fits the distribution to the data returning the parameters defining the Normal distribution.
                By default it estimates the 2 parameters via direct method. The user may fix any of the  parameters by
                initializing the class with that parameter. In this case the fit() will estimate the remainder
                parameter via MLE.

                :param x: a 1D array of data to be fitted.
                :return: a list of the fitted parameters [mu, std].
        """

        params = []

        if method.lower() == 'mle':

            if len(self._not_params_i) == 2:

                params = list(stats.norm.fit(x))

            elif len(self._not_params_i) == 1:
                # [self._not_params_i]  # index of parameters to find. [mean:0, std:1]
                # self._params  # params defined [shape, scale, loc]

                if 0 in self._not_params_i:
                    mu, _ = stats.norm.fit(x, fscale=self._params[1])
                    params = [mu]

                elif 1 in self._not_params_i:
                    _, std = stats.norm.fit(x, floc=self._params[0])
                    params = [std]

        elif method.lower() == 'lsq':
            # 1) compute the empirical distribution function for the presented data:
            x_unique, ecdf_prob = ecdf(x)

            # 2) create error function:
            errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function

            # 3) find parameters:
            res_init = [np.mean(x), np.std(x)]

            params, _ = opt.leastsq(errfunc, res_init, args=(x_unique, ecdf_prob))
            params = list(params)

        else:
            raise ValueError('The method selected is not defined')

        return params


# ----------------------------------- #
#             Log-Normal              #
# ----------------------------------- #
class Lognormal(object):
    def __init__(self, log_mu=None, log_sigma=None):  # mu: scale; sigma: shape
        r"""
        Initialization and fixing the parameters of a log-normal distribution.

        $f(x;\mu,\sigma) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{ln(x-\mu)}{2\sigma^2}}$

        A log-Normal distribution is usually described by two parameters: log-mean, ($\mu$), and log standard deviation,
        ($\sigma$).

        :param mu: Log-Mean.
        :param sigma: Log-Standard Deviation.
        """

        self._params = np.array([log_mu, log_sigma])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

    # ----------------------------------- #
    #                fix                  #
    # ----------------------------------- #
    def fix(self, log_mu=None, log_sigma=None):
        self._params = np.array([log_mu, log_sigma])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

    # ----------------------------------- #
    #                 PDF                 #
    # ----------------------------------- #
    def pdf(self, params, x):
        r"""
        Computes the PDF of Normal distribution. A Normal (Gaussian) distribution is usually described by two
        parameters: mean, ($\mu$), and standard deviation, ($\sigma$).

        $f(x;\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        log_mu = self._params[0]
        log_sigma = self._params[1]

        # Compute and return the PDF for the given x.
        # params in scipy: [shape, loc, scale] -> our (loc is 0, as in RSoft): [log_mu, log_sigma] ->
        # Thus, to compute in scipy: [Log_sigma, 0,  np.exp(log_mu)]
        return stats.lognorm.pdf(x, log_sigma, 0, np.exp(log_mu))

    # ----------------------------------- #
    #                 CDF                 #
    # ----------------------------------- #
    def cdf(self, params, x):
        r"""
        Computes the CDF of a Normal distribution.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        log_mu = self._params[0]
        log_sigma = self._params[1]

        # Compute and return the CDF for the given x.
        # params in scipy: [shape, loc, scale] -> our (loc is 0, as in RSoft): [log_mu, log_sigma] ->
        # Thus, to compute in scipy: [Log_sigma, 0,  np.exp(log_mu)]
        return stats.lognorm.cdf(x, log_sigma, 0, np.exp(log_mu))

    # ----------------------------------- #
    #                 PPF                 #
    # ----------------------------------- #
    def ppf(self, params, q):
        r"""
        Computes the PPF.

        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        log_mu = self._params[0]
        log_sigma = self._params[1]

        # params in scipy: [shape, loc, scale] -> our (loc is 0, as in RSoft): [log_mu, log_sigma] ->
        # Thus, to compute in scipy: [Log_sigma, 0,  np.exp(log_mu)]
        # print([log_sigma, 0, np.exp(log_mu)])
        return stats.lognorm.ppf(q, log_sigma, 0, np.exp(log_mu))

    # ----------------------------------- #
    #              neg_log_lk             #
    # ----------------------------------- #
    def neg_log_lk(self, params, x):
        r"""
        Computes negative sum of log-likelihood for the probability of x given the distribution parameters. This
        function is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is required to convert produtory of
        the Likelihood function in a sum.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        log_mu = self._params[0]
        log_sigma = self._params[1]

        # Compute and return the CDF for the given x.
        # params in scipy: [shape, loc, scale] -> our (loc is 0, as in RSoft): [log_mu, log_sigma] ->
        # Thus, to compute in scipy: [Log_sigma, 0,  np.exp(log_mu)]
        return -np.nansum(stats.lognorm.logpdf(x, log_sigma, 0, np.exp(log_mu)))

    # ----------------------------------- #
    #               log_lk                #
    # ----------------------------------- #
    def log_lk(self, params, x):
        r"""
        Computes sum of log-likelihood for the probability of x given the distribution parameters.
         This function is useful for model validation, i.e. goodness-of-fit. The log transformation is
        required to convert produtory of the Likelihood function in a sum.

        :param params: a list of parameters containing 1) mean and 2) standard deviation.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        log_mu = self._params[0]
        log_sigma = self._params[1]

        # Compute and return the CDF for the given x.
        # params in scipy: [shape, loc, scale] -> our (loc is 0, as in RSoft): [log_mu, log_sigma] ->
        # Thus, to compute in scipy: [Log_sigma, 0,  np.exp(log_mu)]
        return np.nansum(stats.lognorm.logpdf(x, log_sigma, 0, np.exp(log_mu)))

    # ----------------------------------- #
    #                 fit                 #
    # ----------------------------------- #
    def fit(self, x,
            method='MLE'):  # TODO add implementation kwds (if really necessary, or for homogeneity of functions)
        r"""
        Fits the distribution to the data returning the parameters defining the Log-Normal distribution.
                By default it estimates the 2 parameters via direct method. The user may fix any of the  parameters by
                initializing the class with that parameter. In this case the fit() will estimate the remainder
                parameter via MLE.

                :param x: a 1D array of data to be fitted.
                :param method: Maximum Likelihood Estimation, or Least Squares Regression.
                :return: a list of the fitted parameters [log_mu, log_std].
        """

        params = []

        if method.lower() == 'mle':
            if len(self._not_params_i) == 2:

                log_sigma, _, log_mu = list(stats.lognorm.fit(x, floc=0))

                params = [np.log(log_mu), log_sigma]

            elif len(self._not_params_i) == 1:
                # [self._not_params_i]  # index of parameters to find. [mean:0, std:1]
                # self._params  # params defined [shape, scale, loc]

                if 0 in self._not_params_i:
                    _, _, log_mu = stats.lognorm.fit(x, f0=self._params[1], floc=0)
                    params = [np.log(log_mu)]

                elif 1 in self._not_params_i:
                    log_sigma, _, _ = stats.lognorm.fit(x, fscale=np.exp(self._params[0]), floc=0)
                    params = [log_sigma]

        elif method.lower() == 'lsq':
            # 1) compute the empirical distribution function for the presented data:
            x_unique, ecdf_prob = ecdf(x)

            # 2) find parameters:

            res_init = [np.log(x.mean()), np.log(x).std()]

            params, _ = opt.leastsq(self.errfunc, res_init, args=(x_unique, ecdf_prob,)
                                    )
            params = list(params)

            # res = opt.minimize(self.errfunc, res_init, args=(x_unique, ecdf_prob))

            # params = list(res.x)

        else:
            raise ValueError('Method selected is not defined.')

        return params

    def errfunc(self, p, t, y):
        # print(p)
        # print(t)
        # print(y)
        return self.cdf(p, t) - y


# ----------------------------------- #
#            Exponential              #
# ----------------------------------- #
class Exponential(object):
    def __init__(self, scale=None, loc=None):
        r"""
        Initialization and fixing the class of Exponential distribution.

        $f(x;\lambda) = \lambda e^{-\lambda (x-\gamma)}$

        An exponential distribution is usually described by one parameter: $\lambda$ 1/scale. Alternatively, it can use
        the location parameter for shifting the distribution left or right on the x axis.

        :param scale: $\frac{1}{\lambda}$.
        :param loc: $\gamma$.
        """

        self._params = np.array([scale, loc])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                fix                  #
        # ----------------------------------- #

    def fix(self, scale=None, loc=None):
        self._params = np.array([scale, loc])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                 PDF                 #
        # ----------------------------------- #

    def pdf(self, params, x):
        r"""
        Computes the PDF of Normal distribution. A Normal (Gaussian) distribution is usually described by two
        parameters: mean, ($\mu$), and standard deviation, ($\sigma$).

        $f(x;\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        # Compute and return the PDF for the given x.
        return stats.expon.pdf(x, scale=scale, loc=loc)

        # ----------------------------------- #
        #                 CDF                 #
        # ----------------------------------- #

    def cdf(self, params, x):
        r"""
        Computes the CDF of an exponential distribution.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        # Compute and return the PDF for the given x.
        cdf_val = stats.expon.cdf(x, scale=scale, loc=loc)
        # print(cdf_val)
        return cdf_val

        # ----------------------------------- #
        #                 PPF                 #
        # ----------------------------------- #

    def ppf(self, params, q):
        r"""
            Computes the PPF of an exponential distribution.

            """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        return stats.expon.ppf(q, scale=scale, loc=loc)

        # ----------------------------------- #
        #              neg_log_lk             #
        # ----------------------------------- #

    def neg_log_lk(self, params, x):
        r"""
        Computes negative sum of log-likelihood for the probability of x given the distribution parameters. This
        function is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is required to convert produtory of
        the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        # Compute and return the PDF for the given x.
        return -np.nansum(stats.expon.logpdf(x, scale=scale, loc=loc))

        # ----------------------------------- #
        #               log_lk                #
        # ----------------------------------- #

    def log_lk(self, params, x):
        r"""
        Computes sum of log-likelihood for the probability of x given the distribution parameters.
         This function is useful for model validation, i.e. goodness-of-fit. The log transformation is
        required to convert produtory of the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        # Compute and return the PDF for the given x.
        return np.nansum(stats.expon.logpdf(x, scale=scale, loc=loc))

        # ----------------------------------- #
        #                 fit                 #
        # ----------------------------------- #

    def fit(self, x,
            method='MLE'):  # TODO add implementation kwds (if really necessary, or for homogeneity of functions)
        r"""
        Fits the distribution to the data returning the parameters defining the Log-Normal distribution.
                By default it estimates the 2 parameters via direct method. The user may fix any of the  parameters by
                initializing the class with that parameter. In this case the fit() will estimate the remainder
                parameter via MLE.

                :param x: a 1D array of data to be fitted.
                :param method: Maximum Likelihood Estimation, or Least Squares Regression.
                :return: a list of the fitted parameters [scale, loc].
        """

        params = []

        if method.lower() == 'mle':
            if len(self._not_params_i) == 2:

                loc, scale = list(stats.expon.fit(x))

                params = [scale, loc]

            elif len(self._not_params_i) == 1:
                # [self._not_params_i]  # index of parameters to find. [scale:0, loc:1]
                # self._params  # params defined [scale, loc]

                if 0 in self._not_params_i:
                    _, scale = stats.expon.fit(x, floc=self._params[1])
                    params = [scale]

                elif 1 in self._not_params_i:
                    loc, _ = stats.expon.fit(x, fscale=self._params[0])
                    params = [loc]

            else:
                raise ValueError('The number of fixed params is incorrect.')

        elif method.lower() == 'lsq':

            if len(self._not_params_i) == 2:

                bounds_expon = ((1e-6, 1e6), (0, np.min(x),))

                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # # 2) create error function:
                # errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function

                # 3) find parameters:
                loc, scale = stats.expon.fit_loc_scale(x)  # first and second moments
                res_init = [scale, loc]

                # params, = opt.leastsq(errfunc, res_init, args=(x_unique, ecdf_prob))  # since it is unbounded it
                #  converges to an undesirable minimum i.e. loc>min_x
                # params = list(params)

                errfunc = lambda p, t, y: np.sum((self.cdf(p, t) - y) ** 2)  # sse

                res_1 = opt.minimize(errfunc, res_init, args=(x_unique, ecdf_prob),
                                     method='L-BFGS-B',
                                     # method='Nelder-Mead',
                                     bounds=bounds_expon
                                     )

                # print(res_1.success)
                params = list(res_1.x)

            elif len(self._not_params_i) == 1:

                if 0 in self._not_params_i:
                    res_init = [np.mean(x)]

                elif 1 in self._not_params_i:
                    res_init = [0]

                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # 2) create error function:
                errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function

                # 3) find parameters:

                params, _ = opt.leastsq(errfunc, res_init, args=(x_unique, ecdf_prob))
                params = list(params)
            else:
                raise ValueError('The number of parameters is incorrect.')


        else:
            raise ValueError('Method selected is not defined.')

        return params


# ----------------------------------- #
#              Logistic               #
# ----------------------------------- #
class Logistic(object):
    def __init__(self, scale=None, loc=None):
        r"""
        Initialization and fixing the class of Exponential distribution.

        $f(z; \mu, \sigma) = \frac{e^z}{\sigma \cdot (1+e^z)^2} $, where $z = \frac{x-\mu}{\sigma}$

        We are testing ReliaSoft version since it returns improved results. Hence, Log-logistic distribution function is
        characterized by two parameters: Scale, $\sigma$, and Location, $\mu$.

        :param scale: $\mu$.
        :param loc: $\gamma$.
        """

        self._params = np.array([scale, loc])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                fix                  #
        # ----------------------------------- #

    def fix(self, scale=None, loc=None):
        self._params = np.array([scale, loc])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                 PDF                 #
        # ----------------------------------- #

    def pdf(self, params, x):
        r"""
        $f(z; \mu, \sigma) = \frac{e^z}{\sigma \cdot (1+e^z)^2} $, where $z = \frac{x-\mu}{\sigma}$

        Hence, Log-logistic distribution function is characterized by two parameters: Scale, $\sigma$, and Location,
        $\mu$.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        z = (x - loc) / scale
        return np.exp(-z) / (scale * (1 + np.exp(-z)) ** 2)

        # ----------------------------------- #
        #                 CDF                 #
        # ----------------------------------- #

    def cdf(self, params, x):
        r"""
        Computes the CDF of a Logistic distribution.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        # Compute and return the PDF for the given x.
        z = (x - loc) / scale
        return 1 / (1 + np.exp(-z))

        # ----------------------------------- #
        #                 PPF                 #
        # ----------------------------------- #

    def ppf(self, params, q):
        r"""
        Computes the PPF of a Logistic distribution.

        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        scale = self._params[0]
        loc = self._params[1]

        return stats.logistic.ppf(q, loc, scale)

        # ----------------------------------- #
        #              neg_log_lk             #
        # ----------------------------------- #

    def neg_log_lk(self, params, x):
        r"""
        Computes negative sum of log-likelihood for the probability of x given the distribution parameters. This
        function is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is required to convert produtory of
        the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        return -np.nansum(np.log(np.nan_to_num(self.pdf(params, x))))

        # ----------------------------------- #
        #               log_lk                #
        # ----------------------------------- #

    def log_lk(self, params, x):
        r"""
        Computes sum of log-likelihood for the probability of x given the distribution parameters.
         This function is useful for model validation, i.e. goodness-of-fit. The log transformation is
        required to convert produtory of the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        return np.nansum(np.log(np.nan_to_num(self.pdf(params, x))))

        # ----------------------------------- #
        #                 fit                 #
        # ----------------------------------- #

    def fit(self, x, method='MLE',
            implementation=None):  # TODO add implementation kwds (if really necessary, or for homogeneity of functions)
        r"""
        Fits the distribution to the data returning the parameters defining the Logistic distribution.
                By default it estimates the 2 parameters via direct method. The user may fix any of the  parameters by
                initializing the class with that parameter. In this case the fit() will estimate the remainder
                parameter via MLE.

                :param x: a 1D array of data to be fitted.
                :param method: Maximum Likelihood Estimation, or Least Squares Regression.
                :param implementation: Implementation type. Either NANO, or SCIPY.
                :return: a list of the fitted parameters [scale, loc].
        """

        params = list()

        if len(self._not_params_i) <= 2:  # if the initialization was made with no fixed parameters
            init_params = np.array([1, np.mean(x), np.min(x)])[self._not_params_i]
            bounds_logistic = np.array(((1e-6, 1e10), (-1e10, 1e10)), dtype=float)

            if implementation is None:
                implementation = 'nano'

            if method.lower() == 'mle':
                if implementation.lower() == 'nano':
                    loc, scale = stats.logistic.fit_loc_scale(x)  # first and second moments
                    res_init = [scale, loc]

                    res_1 = opt.minimize(self.neg_log_lk, res_init, args=(x,),
                                         method='L-BFGS-B',
                                         # method='Nelder-Mead',
                                         bounds=bounds_logistic
                                         )

                    # print(res_1.success)
                    params = list(res_1.x)

                if implementation.lower() == 'scipy':
                    loc, scale = stats.logistic.fit(x)
                    params = [scale, loc]

            elif method.lower() == 'lsq':  # TODO test this
                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # 2) create error function:
                errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function

                # 3) find parameters:
                # print(bounds_logistic)
                # res_init = opt.differential_evolution(self.neg_log_lk, args=(x,),
                #                                       bounds=bounds_logistic
                #                                       )
                loc, scale = stats.logistic.fit_loc_scale(x)  # first and second moments
                res_init = [scale, loc]

                params, _ = opt.leastsq(errfunc, res_init, args=(x_unique, ecdf_prob))
                params = list(params)

            else:
                raise ValueError('Method selected is not defined.')


        else:
            raise ValueError('Unacceptable number of parameters.')

        return params


# ----------------------------------- #
#            Log-Logistic             #
# ----------------------------------- #
class Loglogistic(object):
    def __init__(self, shape=None, scale=None):
        r"""
        Initialization and fixing the class of Log-logistic distribution, AKA Fisk distribution.

        $f(z; shape, scale) = \frac{e^z}{\sigma \cdot x \cdot (1+e^z)^2} $, where $z = \frac{ln(x)-scale}{shape}$

        We are testing ReliaSoft version since it returns improved results. Hence, Log-logistic distribution function is
        characterized by two parameters: Scale, $\sigma$, and Location, $\mu$.

        :param shape:
        :param scale:
        """

        self._params = np.array([shape, scale])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                fix                  #
        # ----------------------------------- #

    def fix(self, shape=None, scale=None):
        self._params = np.array([shape, scale])
        self._not_params_i = [i for i, f in enumerate(self._params) if f is None]

        # ----------------------------------- #
        #                 PDF                 #
        # ----------------------------------- #

    def pdf(self, params, x):
        r"""
        Probability Density Function (PDF) of log-logistic distribution.

        :param params: a list of parameters containing 1) shape and 2) scale.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Probability of x given the distribution parameters.
        """

        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        shape = self._params[0]
        scale = self._params[1]

        z = (np.log(x) - scale) / shape
        return np.exp(-z) / (shape * x * (1 + np.exp(-z)) ** 2)

        # ----------------------------------- #
        #                 CDF                 #
        # ----------------------------------- #

    def cdf(self, params, x):
        r"""
        Computes the CDF of a Logistic distribution.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        shape = self._params[0]
        scale = self._params[1]

        # Compute and return the PDF for the given x.
        z = (np.log(x) - scale) / shape
        return 1 / (1 + np.exp(-z))

        # ----------------------------------- #
        #                 PPF                 #
        # ----------------------------------- #

    def ppf(self, params, q):
        r"""
        Computes the PPF of a Loglogistic distribution.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param q: a number, a list, or an array of x values for computing probability.
        :return: Cumulative Probability for x given the distribution parameters.
        """
        if len(params) == len(self._not_params_i):
            self._params[self._not_params_i] = params
        else:
            print('Len params: {}'.format(len(params)))
            print('Len self._params: {}'.format(len(self._params)))
            raise ValueError('The size of parameters is not correct.')

        shape = self._params[0]
        scale = self._params[1]

        return stats.fisk.ppf(q, 1 / shape, 0, np.exp(scale))

        # ----------------------------------- #
        #              neg_log_lk             #
        # ----------------------------------- #

    def neg_log_lk(self, params, x):
        r"""
        Computes negative sum of log-likelihood for the probability of x given the distribution parameters. This
        function is useful as a cost function for optimization problem regarding Maximum
        Likelihood Estimation (MLE) of the parameters. The log transformation is required to convert productory of
        the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        return -np.nansum(np.log(np.nan_to_num(self.pdf(params, x))))

        # ----------------------------------- #
        #               log_lk                #
        # ----------------------------------- #

    def log_lk(self, params, x):
        r"""
        Computes sum of log-likelihood for the probability of x given the distribution parameters.
         This function is useful for model validation, i.e. goodness-of-fit. The log transformation is
        required to convert produtory of the Likelihood function in a sum.

        :param params: a list of parameters containing 1) scale and 2) location.
        :param x: a number, a list, or an array of x values for computing probability.
        :return: negative sum of log-likelihood for the probability of x given the distribution parameters.
        """

        return np.nansum(np.log(np.nan_to_num(self.pdf(params, x))))

        # ----------------------------------- #
        #                 fit                 #
        # ----------------------------------- #

    def fit(self, x, method='MLE',
            implementation=None):  # TODO add implementation kwds (if really necessary, or for homogeneity of functions)
        r"""
        Fits the distribution to the data returning the parameters defining the Logistic distribution.
                By default it estimates the 2 parameters via direct method. The user may fix any of the  parameters by
                initializing the class with that parameter. In this case the fit() will estimate the remainder
                parameter via MLE.

                :param x: a 1D array of data to be fitted.
                :param method: Maximum Likelihood Estimation, or Least Squares Regression.
                :param implementation: Implementation type. Either NANO, or SCIPY.
                :return: a list of the fitted parameters [scale, loc].
        """

        params = list()

        if len(self._not_params_i) <= 2:  # if the initialization was made with no fixed parameters
            init_params = np.array([1, np.mean(x), np.min(x)])[self._not_params_i]
            bounds_logistic = ((1e-6, 1e10), (-1e10, 1e10))

            if implementation is None:
                implementation = 'nano'

            if method.lower() == 'mle':
                if implementation.lower() == 'nano':
                    res_init = opt.differential_evolution(self.neg_log_lk, args=(x,),
                                                          bounds=bounds_logistic
                                                          )

                    # TODO review the initialization parameters. res_init.x was slow but working.
                    res_1 = opt.minimize(self.neg_log_lk, res_init.x, args=(x,),
                                         method='L-BFGS-B',
                                         # method='Nelder-Mead',
                                         bounds=bounds_logistic
                                         )

                    # print(res_1.success)
                    params = list(res_1.x)

                if implementation.lower() == 'scipy':
                    shape, _, scale = stats.fisk.fit(x, floc=0)  # first and second moments
                    params = [1 / shape, np.log(scale)]  # alternative form of representation (i.e. standardized)

            elif method.lower() == 'lsq':
                # 1) compute the empirical distribution function for the presented data:
                x_unique, ecdf_prob = ecdf(x)

                # 2) create error function:
                errfunc = lambda p, t, y: (self.cdf(p, t) - y)  # Distance to the target function

                # 3) find parameters:
                # print(bounds_logistic)
                # res_init = opt.differential_evolution(self.neg_log_lk, args=(x,),
                #                                       bounds=bounds_logistic
                #                                       )
                # loc, scale = stats.logistic.fit_loc_scale(x)  # first and second moments
                # res_init = [scale, loc]
                res_init = [1, 1]

                params, _ = opt.leastsq(errfunc, res_init, args=(x_unique, ecdf_prob))
                params = list(params)

            else:
                raise ValueError('Method selected is not defined.')

        else:
            raise ValueError('Unacceptable number of parameters.')

        return params
