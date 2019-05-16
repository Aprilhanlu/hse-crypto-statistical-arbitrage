import datetime
import requests
import time

import numpy as np


OPEN_PRICE_INDEX = 3
COINTEGRATION_WEIGHTS = np.array([0.00838272, -0.38797855,  0.95798562, -0.25152476])
COINTEGRATION_SET = ["ETH-USD", "BTC-USD", "BCH-USD", "LTC-USD"]
CURRENCIES_MIN_AMOUNT = {"ETH-USD": 0.01, "BTC-USD": 0.001,
                         "BCH-USD": 0.01, "LTC-USD": 0.1}
STATS_CALC_WINDOW_SIZE = 500
Z_SCORE_OPEN = 3
Z_SCORE_CLOSE = 1


def get_data(kind_of_data, pair, granularity, start_date, end_date):
    url_base = "https://api.pro.coinbase.com/products/"
    start_date += "-0000"
    end_date += "-0000"
    data = requests.get(url_base + pair + "/" + kind_of_data + "?granularity=" + str(granularity) + \
                        "&start=" + start_date + "&end=" + end_date)

    return data.json()


current_time = datetime.datetime.utcnow().replace(second=0, microsecond=0)
assets_historical_rates = dict()
for asset_name in COINTEGRATION_SET:
    start_time = current_time - datetime.timedelta(minutes=300)
    data = get_data("candles", asset_name, 60, start_time.isoformat(), current_time.isoformat())
    data = np.array(data)
    assets_historical_rates[asset_name] = data[:300, OPEN_PRICE_INDEX]
    time.sleep(0.3)
last_time = current_time

while True:
    coint_series = assets_historical_rates[COINTEGRATION_SET[0]] * CURRENCIES_MIN_AMOUNT[COINTEGRATION_SET[0]]
    coint_series *= COINTEGRATION_WEIGHTS[0]
    for asset_index in range(1, len(COINTEGRATION_SET)):
        asset_name = COINTEGRATION_SET[asset_index]
        coint_weight = COINTEGRATION_WEIGHTS[asset_index]
        min_amount = CURRENCIES_MIN_AMOUNT[asset_name]
        coint_series += assets_historical_rates[asset_name] * min_amount * coint_weight

    coint_series_mean = np.nanmean(coint_series[-STATS_CALC_WINDOW_SIZE:])
    coint_series_std = np.nanstd(coint_series[-STATS_CALC_WINDOW_SIZE:])
    last_value = coint_series[-1]
    z_score = (last_value - coint_series_mean) / coint_series_std + 2

    print(datetime.datetime.utcnow(), "||", z_score, "||", end=" ")
    if abs(z_score) >= Z_SCORE_OPEN:
        capital_allocation = np.array([assets_historical_rates[asset_name][-1] for asset_name in COINTEGRATION_SET])
        capital_allocation *= np.array(list(CURRENCIES_MIN_AMOUNT.values()))
        capital_allocation *= COINTEGRATION_WEIGHTS
        capital_allocation /= np.abs(capital_allocation).sum()
        if z_score > 0:
            capital_allocation *= -1
            capital_allocation = {asset_name: str(round(capital_allocation[asset_index] * 100, 2)) + "%" for asset_index, asset_name in enumerate(COINTEGRATION_SET)}
            print("Open positions " + str(capital_allocation))
        else:
            capital_allocation = {asset_name: str(round(capital_allocation[asset_index] * 100, 2)) + "%" for
                                  asset_index, asset_name in enumerate(COINTEGRATION_SET)}
            print("Open positions " + str(capital_allocation))
    elif abs(z_score) <= Z_SCORE_CLOSE:
        print("Close positions")
    else:
        print("Close positions")

    time.sleep(60)
    current_time = datetime.datetime.utcnow().replace(second=0, microsecond=0)
    for asset_name in COINTEGRATION_SET:
        data = get_data("candles", asset_name, 60, last_time.isoformat(), current_time.isoformat())
        data = np.array(data)
        assets_historical_rates[asset_name] = np.hstack([assets_historical_rates[asset_name], data[:, OPEN_PRICE_INDEX]])
        time.sleep(0.3)
    last_time = current_time

