import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error

class TimeSeriesStationarizer:
    def __init__(self, df, window=12, signif=0.05):
        """
        df: pandas DataFrame of time series features (datetime index recommended)
        window: rolling window size for Z-score normalization
        signif: significance level for ADF test
        """
        self.df = df.copy()
        self.window = window
        self.signif = signif
        self.df_stationary = pd.DataFrame(index=df.index)
        self.results = pd.DataFrame(columns=['ADF_p_orig', 'Stationary_orig',
                                             'ADF_p_stn', 'Stationary_stn'])

    def adf_test(self, series):
        """
        Returns p-value and stationarity boolean
        """
        series = series.dropna()
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        stationary = p_value < self.signif
        return p_value, stationary

    def rolling_zscore(self, series):
        rolling_mean = series.rolling(window=self.window, min_periods=1).mean()
        rolling_std = series.rolling(window=self.window, min_periods=1).std()
        return (series - rolling_mean) / rolling_std

    def fit_transform(self):
        print(f"{'Feature':20s} | {'ADF p (orig)':12s} | {'Stationary?':10s} | {'ADF p (stn)':12s} | {'Stationary?'}")
        print("-"*80)
        
        for col in self.df.columns:
            p_orig, stationary_orig = self.adf_test(self.df[col])
            
            if stationary_orig:
                self.df_stationary[col] = self.df[col]
                p_stn, stationary_new = p_orig, stationary_orig
            else:
                self.df_stationary[col] = self.rolling_zscore(self.df[col])
                p_stn, stationary_new = self.adf_test(self.df_stationary[col])
            
            self.results.loc[col] = [p_orig, stationary_orig, p_stn, stationary_new]
            
            print(f"{col:20s} | {p_orig:<12.4f} | {str(stationary_orig):<10s} | {p_stn:<12.4f} | {stationary_new}")
        
        return self.df_stationary

    # -----------------------------
    # Example usage
    # -----------------------------
    # df = pd.read_csv("your_timeseries.csv", index_col=0, parse_dates=True)
    # ts_stationarizer = TimeSeriesStationarizer(df, window=12)
    # df_stationary = ts_stationarizer.fit_transform()
    # print(ts_stationarizer.results)




class WalkForwardCV:
    """
    Simple walk-forward cross-validation using sklearn's TimeSeriesSplit.

    Parameters
    ----------
    model : sklearn estimator
        Must have fit() and predict() methods.
    n_splits : int
        Number of folds.
    scoring : callable, optional
        Custom scoring function (y_true, y_pred) -> float.
    gap : int, default=0
        Number of samples to exclude between train and test to prevent leakage.
    """

    def __init__(self, model, n_splits=5, scoring=None, gap=0):
        self.model = model
        self.n_splits = n_splits
        self.scoring = scoring or (lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred))
        self.gap = gap

    def evaluate(self, X, y, verbose=True):
        """Perform walk-forward CV and return fold scores.
        Parameters
        ----------
         X : pandas.DataFrame or numpy.ndarray
            Feature matrix (time-ordered, not shuffled).
            Shape: (n_samples, n_features)
            Must be aligned with `y`.

        y : pandas.Series, numpy.ndarray, or list-like
            Target variable corresponding to X.
            Shape: (n_samples,)
        """
         
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = clone(self.model)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = self.scoring(y_test, preds)
            scores.append(score)

            if verbose:
                print(f"Fold {fold+1}: {score:.4f}")

        print(f"Average score: {np.mean(scores):.4f}")
        return scores
    
    # Example usage:
    # X = np.random.randn(1000, 10)
    # y = np.random.randn(1000)

    # # Custom metric: Spearman correlation
    # def spearman_corr(y_true, y_pred):
    #     return spearmanr(y_true, y_pred).correlation

    # # Instantiate and run
    # cv = WalkForwardCV(model=RandomForestRegressor(), n_splits=5, scoring=spearman_corr)
    # scores = cv.evaluate(X, y)