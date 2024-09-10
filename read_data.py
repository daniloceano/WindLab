import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("/home/daniloceano/Documents/Programs_and_scripts/readwindcube/WLS866-104_2024_08_01__00_00_00.rtd",
                 encoding='unicode_escape',  skiprows=range(41), sep='\t')

def compute_correlation_variation_detrended_data(data):
    rolling_mean = data.rolling(window=600, min_periods=1).mean()
    detrended_data = data - rolling_mean
    std_detrended_data = np.std(detrended_data)
    # coef_variation_detrended_data = np.std(detrended_data) / np.mean(detrended_data)
    return std_detrended_data

ws40m = df['40m Wind Speed (m/s)']
std_detrended_ws40m = compute_correlation_variation_detrended_data(ws40m)
# autocorr = []
# for i in range(1000):
#     autocorr.append(ws40m.autocorr(lag=i))
# plt.figure()
# plt.plot(autocorr)
# plt.axhline(y=0.3, color='r', linestyle='-')

ws60m = df['60m Wind Speed (m/s)']
std_detrended_ws60m = compute_correlation_variation_detrended_data(ws60m)

ws70m = df['70m Wind Speed (m/s)']
std_detrended_ws70m = compute_correlation_variation_detrended_data(ws70m)

ws100m = df['100m Wind Speed (m/s)']
std_detrended_ws100m = compute_correlation_variation_detrended_data(ws100m)

ws120m = df['120m Wind Speed (m/s)']
std_detrended_ws120m = compute_correlation_variation_detrended_data(ws120m)

ws150m = df['150m Wind Speed (m/s)']
std_detrended_ws150m = compute_correlation_variation_detrended_data(ws150m)

time = pd.to_datetime(df['Timestamp'])

print(time)

# plt.figure()
# plt.plot(time, windir40m)
# plt.show()