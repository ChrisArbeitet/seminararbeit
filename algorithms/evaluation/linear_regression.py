from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from typing import Union
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import t

"""
module provides Regressor class with different linear regression modalities to fit models and evaluate them
"""

class Regressor:
    """
    Initialization, training, testing, and evaluation of OLS regressions.

    Parameters
    ----------
    target_column: str
        Target of the evaluation for addressing the predicted values in OLS.
    Methods
    -------
    perform_ols:
        Initialization, training, testing, and evaluation of OLS model.
    regression_evaluation:
        Evaluate a given model on different properties.
    """

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self.predicted_values = pd.DataFrame
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def perform_ols(self) -> Union[None, pd.Series]:
        system_ols = LinearRegression()

        system_ols.fit(self.x_train, self.y_train)

        predicted_value = system_ols.predict(self.x_test)

        return system_ols, predicted_value

    def perform_pls(self, n_components: int = 2) -> Union[None, pd.Series]:
        system_pls = PLSRegression(n_components=n_components)

        system_pls.fit(self.x_train, self.y_train)

        predicted_value = system_pls.predict(self.x_test)

        return system_pls, predicted_value

    def perform_ridge(self, alpha: float = 1.0) -> Union[None, pd.Series]:
        system_ridge = Ridge(alpha=alpha)

        system_ridge.fit(self.x_train, self.y_train)

        predicted_value = system_ridge.predict(self.x_test)

        return system_ridge, predicted_value

    def regression_evaluation(self, actual_values: list, predicted_values: np.array, model) -> pd.Series:
        """
        Evaluate a given model on different properties.

        Parameters
        ----------
        actual_values: list
            Actual values of the target of the regression.
        predicted_values: array
            Predicted values from the model of the target.
        model: object

        Return
        ------------
        evaluation_results_df: DataFrame
            DataFrame with the produced properties of the model after evaluation.

        Notes
        -----
        Reserve disabled cause reserve has to be generated fitting to given data.
        Method of generation is not defined yet.
        """
        if model is None:
            # if mistake in given data for model, skip model evaluation and assign individual a very high score
            print("No Model to evaluate. Mistake in data.")
            return pd.Series({'Mittlerer Fehler [%]': 1000,
                              'Mittlerer absoluter Fehler [%]': 1000,
                              'Mittlerer maximaler absoluter Fehler [%]': 1000,
                              'Wurzel der mittleren quadratischen Abweichung [%]': 1000,
                              'Notwendige Reserven [%]': 1000,
                              'Signifikanzniveau zur Korrelation [%]': 1000})
        y = np.array(actual_values)
        y_hat = np.array(predicted_values)
        mean_y = np.mean(y)
        # number of test values
        n = y.shape[0]

        mean_error_prozent = np.mean(y_hat - y) / mean_y
        mean_absolute_error_prozent = np.mean(abs(y_hat - y)) / mean_y
        max_absolute_error_prozent = np.max(abs(y_hat - y)) / mean_y
        root_mean_squared_error_prozent = (np.sqrt(np.sum((y - y_hat) ** 2)) / (n - 1)) / mean_y

        # correlation between y and y_hat
        r, p_value = pearsonr(y.ravel(), y_hat.ravel())
        # t-test
        t_value = np.abs(r) * np.sqrt((n - 2) / (1 - r ** 2))
        #  two-tailed p-value, from excel code:1-T.DIST.2T(B54,B52)  t-statistic (B54) and degrees of freedom (B52) of
        # evaluation excel file
        significance_level_correlation = 2 * (1 - stats.t.cdf(abs(t_value), df=n))
        if significance_level_correlation == p_value:
            print("same significance")
        # first attempt on reserve
        # prediction error
        error = y_hat - y
        #  standard deviation of the error
        std_error = np.std(error, ddof=1)  # bessel correction sample
        # determine the appropriate multiple for a 95% confidence interval (one-sided)
        t_multiplier = t.ppf(0.95, df=len(y) - 1)
        # Calculate the necessary reserve as a multiple of the standard deviation of the error
        necessary_reserves = t_multiplier * std_error

        evaluation_results_df = pd.Series({'Mittlerer Fehler [%]': abs(mean_error_prozent) * 100,
                                           'Mittlerer absoluter Fehler [%]': abs(mean_absolute_error_prozent) * 100,
                                           'Mittlerer maximaler absoluter Fehler [%]': abs(
                                               max_absolute_error_prozent) * 100,
                                           'Wurzel der mittleren quadratischen Abweichung [%]': abs(
                                               root_mean_squared_error_prozent) * 100,
                                           'Notwendige Reserven [%]': abs(necessary_reserves)/mean_y * 100,
                                           'Signifikanzniveau zur Korrelation [%]': significance_level_correlation})

        return evaluation_results_df
