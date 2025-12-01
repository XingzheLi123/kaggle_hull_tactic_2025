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



from sklearn.decomposition import KernelPCA
class FeatureEnricher:
    """
    Enriches time series data with lags, rolling stats, time-based features, and transformations.
    """
    def __init__(self, 
                 lags=[1, 2, 5, 10], 
                 rolling_windows=[5, 10, 20], 
                 add_lags=True,
                 add_rolling=True,
                 add_diff=False,
                 add_pct_change=False):
        """
        Parameters
        ----------
        lags : list[int]
            List of lag periods to include.
        rolling_windows : list[int]
            List of window sizes for rolling statistics.
        use_cyclical_time : bool
            Whether to add cyclical time features (sin/cos of day/month).
        add_diff : bool
            Whether to include first differences.
        add_pct_change : bool
            Whether to include percent change.
        """
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.add_lags = add_lags
        self.add_rolling = add_rolling
        self.add_diff = add_diff
        self.add_pct_change = add_pct_change
    def add_lags_features(self, df, col):
        """
        col : list[str]
        """
        lag_features = {
            f"{col}_lag{lag}": df[col].shift(lag)
            for lag in self.lags
        }
        return pd.DataFrame(lag_features, index=df.index)
    
    def add_rolling_features(self, df, col):
        roll_features = {}
        for w in self.rolling_windows:
            roll_features[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            roll_features[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
        return pd.DataFrame(roll_features, index=df.index)
    
    def add_differences_features(self, df, col):
        diff_features = {}
        if self.add_diff:
            diff_features[f"{col}_diff1"] = df[col].diff(1)
        if self.add_pct_change:
            diff_features[f"{col}_pctchg"] = df[col].pct_change()
        return pd.DataFrame(diff_features, index=df.index)
    
    def transform(self, df, target_cols):
        """
        target_cols : list[str]
            Feature columns to enrich
        """
        df = df.copy()
        feature_frames = [df[target_cols]]
        for col in target_cols:
            if self.add_lags:
                feature_frames.append(self.add_lags_features(df, col))
            if self.add_rolling:
                feature_frames.append(self.add_rolling_features(df, col))
            feature_frames.append(self.add_differences_features(df, col))
            
        result = pd.concat(feature_frames, axis=1)
        # Drop rows with NaNs introduced by shifting/rolling
        result = result.dropna()
        return result
    
    # -----------------------------
    # Example usage
    # -----------------------------
    # df = pd.read_csv("your_timeseries.csv")
    # ts_enricher = FeatureEnricher(add_diff=True, add_pct_change=True)
    # train_df_enriched = ts_enricher.transform(train_df, standard_features)
    

