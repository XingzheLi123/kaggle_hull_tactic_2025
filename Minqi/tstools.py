import pandas as pd
from statsmodels.tsa.stattools import adfuller

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