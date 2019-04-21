import itertools
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import statsmodels.api as sm
from collections import defaultdict
from tqdm import tqdm_notebook
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def transform_coint_weights(weights):
    weights /= np.sum(np.absolute(weights))


def list_market(candles_folder_path, currency, is_print=True):
    pairs = os.listdir(candles_folder_path)
    selected_pairs = []
    for pair in pairs:
        if pair.endswith(f"{currency}.csv"):
            selected_pairs.append(pair)
            if is_print:
                print(pair)

    return selected_pairs


def file_names_to_tickers(file_names):
    return list(map(lambda x: x[:-4], file_names))


def combine_market_data(folder, market_name, column, is_cut_nas=False, file_names_to_exclude=None, is_interpolate=True):
    file_names_to_exclude = file_names_to_exclude or []
    market = list_market(folder, market_name, False)
    market = [file_name for file_name in market if file_name not in file_names_to_exclude]
    serieses = []
    print(market)
    for pair in market:
        pair_data = pd.read_csv(f"{folder}{pair}", index_col="time", parse_dates=["time"])
        serieses.append(pair_data[column])

    ticker_names = file_names_to_tickers(market)
    data = pd.concat(serieses, axis=1, join="outer")
    data.columns = ticker_names

    if is_cut_nas:
        data.dropna(inplace=True)

    data = data.resample("1min").mean()

    if is_interpolate:
        data = data.interpolate(method="linear")

    return data


def create_zipline_panel(files_path, pairs, min_sizes):

    if not pairs[0].endswith(".csv"):
        pairs_files = map(lambda x: x + ".csv", pairs)
    data = dict()
    for pair_file, pair_name in zip(pairs_files, pairs):
        pair_data = pd.read_csv(f"{files_path}{pair_file}", index_col="time", parse_dates=["time"])
        pair_data.index.name = "date"
        pair_data = pair_data.resample("1min").mean()
        pair_data.loc[:, "volume"] = round(pair_data.loc[:, "volume"] / min_sizes[pair_name])
        pair_data.loc[:, ["open", "high", "low", "close"]] *= min_sizes[pair_name]
        data[pair_name] = pair_data

    panel = pd.Panel(data)
    del data
    panel.minor_axis = ["open", "high", "low", "close", "volume"]
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)

    return panel


def find_pairs(prices, coint_set_amount, johansen_lag):
    # Check all pairs inside one cluster for cointegration
    result = {}
    traded_assets = prices.columns
    total_pairs = 1
    for i in range(coint_set_amount):
        total_pairs *= (len(traded_assets) - i)
    for i in range(1, coint_set_amount + 1):
        total_pairs /= i
    with tqdm_notebook(total=total_pairs) as pbar:
        for combination in itertools.combinations(traded_assets, coint_set_amount):
            combination_prices = prices[list(combination)]
            johansen_result = coint_johansen(combination_prices.values, det_order=0, k_ar_diff=johansen_lag)
            weights = johansen_result.lr1.reshape(-1, 1) >= johansen_result.cvt
            weights = weights.any(axis=1)
            weights = johansen_result.evec[weights]
            if (johansen_result.lr1.reshape(-1, 1) >= johansen_result.cvt).any():
                result[combination] = weights[0]
            pbar.update(1)

    return result


# def estimate_pairs_z_score(data, pairs_coint_weights, pairs_to_calc, window_size):
#     pairs_z_scores = dict()
#     for pair in pairs_to_calc:
#         weights = pairs_coint_weights[pair]
#         pair_sids = list(map(lambda x: zipline.api.symbol(x), pair))
#         pair_values = data.history(pair_sids, fields="price", bar_count=window_size,
#                                    frequency="1m")
#         print(pair_values.iloc[0])
#         pair_values = (pair_values * weights).sum(axis=1)
#         pair_values.interpolate(limit_direction="both", inplace=True)
#         z_score = (pair_values[-1] - pair_values.mean()) / pair_values.std()
#         pairs_z_scores[pair] = z_score
#
#     return pairs_z_scores


def filter_pairs_by_z_score(context, data, pairs_z_scores):
    pairs_z_scores = pairs_z_scores.items()
    pairs_z_scores = filter(lambda x: abs(x[1]) >= context.Z_SCORE_OPEN, pairs_z_scores)
    pairs_z_scores = dict(pairs_z_scores)

    return pairs_z_scores


def select_pairs(pairs_z_scores, pairs_coint_weights, z_score_open):
    eligble_pairs = pairs_z_scores.keys()
    eligble_pairs = filter(lambda x: abs(pairs_z_scores[x]) >= z_score_open, eligble_pairs)

    assets_sign = dict()
    pairs_for_trading = list()
    for pair in eligble_pairs:
        sign = 1
        if pairs_z_scores[pair] > 0:
            sign = -1
        is_select_pair = True
        for asset_idx, asset in enumerate(pair):
            asset_sign = -1 if pairs_coint_weights[pair][asset_idx] * sign < 0 else 1
            if assets_sign.get(asset, None) is not None and assets_sign[asset] != asset_sign:
                is_select_pair = False
                break
        if is_select_pair:
            for asset_idx, asset in enumerate(pair):
                asset_sign = -1 if pairs_coint_weights[pair][asset_idx] * sign < 0 else 1
                if assets_sign.get(asset, None) is None:
                    assets_sign[asset] = asset_sign
            pairs_for_trading.append(pair)

    return pairs_for_trading


def capital_weights(pairs_z_scores, selected_pairs, pairs_coint_weights):
    selected_pairs_z_score_abs_sum = 0
    for pair in selected_pairs:
        selected_pairs_z_score_abs_sum += abs(pairs_z_scores[pair])

    asset_in_pair_weights = dict()
    assets_weights = defaultdict(float)
    for pair in selected_pairs:
        pair_weight = -pairs_z_scores[pair] / selected_pairs_z_score_abs_sum
        pair_coint_weights = pairs_coint_weights[pair]
        pair_coint_weights /= np.sum(np.absolute(pair_coint_weights))
        asset_in_pair_weights[pair] = pair_coint_weights * pair_weight
        for asset_idx, asset in enumerate(pair):
            assets_weights[asset] += asset_in_pair_weights[pair][asset_idx]

    for pair in selected_pairs:
        for asset_idx, asset in enumerate(pair):
            asset_in_pair_weights[pair][asset_idx] /= abs(assets_weights[asset])

    return assets_weights, asset_in_pair_weights


# def adjust_positions_by_target_percent(context, data):
#     for stock, target_percent in context.traded_stocks_target_percent.items():
#         if target_percent != 0:
#             zipline.api.order_target_percent(stock, target_percent)


def draw_cointegration(pair, data, selected_weights, window_size, start_date, end_date):
    plt.figure(figsize=(20, 10))

    pair_value = (data[list(pair)] * selected_weights).sum(axis=1)
    mean_values = pair_value.rolling(window_size).mean()
    std_values = pair_value.rolling(window_size).std()

    pair_value = pair_value.loc[start_date:end_date]
    mean_values = mean_values.loc[start_date:end_date]
    std_values = std_values.loc[start_date:end_date]

    std_up = mean_values + 2 * std_values
    std_down = mean_values - 2 * std_values

    pair_value.plot(label="Cointegrating Combination")
    std_up.plot(label="2 std above the mean", style="--", alpha=1)
    std_down.plot(label="2 std below the mean", style="--", alpha=1)
    plt.title(f"{pair} cointegrating combination")
    mean_values.plot(label="1 Hour Rolling Mean")
    plt.rcParams.update({'font.size': 20})
    plt.legend(prop={'size': 15})
    plt.grid()
    plt.show()


def research_pair_trading_opportunity(currency1, currency2):
    name1, name2 = currency1.name, currency2.name
    print(f"Researching Pair {name1} and {name2}")
    model = OLS(currency1, sm.add_constant(currency2))
    ols_results = model.fit()
    print("Prices OLS results:")
    print(f"const: {ols_results.params['const']} || {name2} {ols_results.params[name2]}")

    coint_series = currency1 - currency2 * ols_results.params[name2]
    coint_series.plot()
    plt.show()

    dependent_var = coint_series.diff()[1:]
    independent_var = coint_series.shift(1)[1:]
    independent_var.name = "val_prev"
    model = OLS(dependent_var, sm.add_constant(independent_var))
    ols_results = model.fit()
    ols_results.params

    print("Diff of Cointegrating Series OLS Results:")
    print(f"const: {ols_results.params['const']} || {ols_results.params['val_prev']}")
    print("Mean-Reverse Half-life:", -np.log(2) / ols_results.params["val_prev"])


def check_serieses_pairs_for_cointegration(data, is_print_test_statistic=True, is_print_p_value=True,
                                           is_print_critical_value=True):
    for currencies_combination in combinations(data.columns.values, 2):
        pair1 = currencies_combination[0]
        pair2 = currencies_combination[1]
        coint_results = coint(data[pair1], data[pair2], maxlag=1)
        print("--------------------------")
        print(f"Pair {pair1} and {pair2}")
        if is_print_test_statistic:
            print("Test Statistic:", coint_results[0])
        if is_print_p_value:
            print("P-Value:", coint_results[1])
        if is_print_critical_value:
            print("Critical values:", coint_results[2])


def upload_currency_data(path, start_date, end_date):
    currency = pd.read_csv(path, index_col="time", parse_dates=["time"])
    currency = currency.loc[start_date:end_date]
    currency["usd_vol"] = currency["close"] * currency["volume"]
    currency = currency.resample("min").mean()

    return currency


def sample_pair_as_dollar_bars(currency1_path, currency2_path, start_date, end_date, dollar_bar_size):
    currency1_data = upload_currency_data(currency1_path, start_date, end_date)
    currency2_data = upload_currency_data(currency2_path, start_date, end_date)

    currencies_new_data = [[["time_start", "time_end", "open", "high", "low", "close", "volume", "usd_vol"]],
                           [["time_start", "time_end", "open", "high", "low", "close", "volume", "usd_vol"]]]
    bar_start, bar_end = None, None
    current_usd_vol_traded = 0
    open_prices, high_prices, low_prices, close_prices, volumes = [], [-np.inf] * 2, [np.inf] * 2, [], [0] * 2
    dollar_volumes = [0] * 2
    is_creating_bar = False
    for index in tqdm_notebook(currency1_data.index):
        if index > currency1_data.index[-1] or index > currency2_data.index[-1]:
            break
        currencies = currency1_data.loc[index], currency2_data.loc[index]
        if not is_creating_bar and not np.isnan(currencies[0]["open"]) and not np.isnan(currencies[1]["open"]):
            is_creating_bar = True
            bar_start = index
            for currency in currencies:
                open_prices.append(currency["open"])
        if is_creating_bar:
            for currency_index, currency in enumerate(currencies):
                if not np.isnan(currency["high"]):
                    high_prices[currency_index] = max(high_prices[currency_index], currency["high"])
                if not np.isnan(currency["low"]):
                    low_prices[currency_index] = min(low_prices[currency_index], currency["low"])
                if not np.isnan(currency["volume"]):
                    volumes[currency_index] += currency["volume"]
                if not np.isnan(currency["usd_vol"]):
                    current_usd_vol_traded += currency["usd_vol"]
                    dollar_volumes[currency_index] += currency["usd_vol"]
            if current_usd_vol_traded > dollar_bar_size and not np.isnan(currencies[0]["close"]) and\
                    not np.isnan(currencies[1]["close"]):
                bar_end = index
                for currency in currencies:
                    close_prices.append(currency["close"])
                for currency_index in range(2):
                    new_row = [bar_start, bar_end, open_prices[currency_index], high_prices[currency_index],
                               low_prices[currency_index], close_prices[currency_index], volumes[currency_index],
                               dollar_volumes[currency_index]]
                    currencies_new_data[currency_index].append(new_row)
                is_creating_bar = False
                current_usd_vol_traded = 0
                open_prices, high_prices = [], [-np.inf] * 2
                low_prices, close_prices, volumes = [np.inf] * 2, [], [0] * 2
                dollar_volumes = [0] * 2

    currency1_df = pd.DataFrame(currencies_new_data[0][1:], columns=currencies_new_data[0][0])
    currency2_df = pd.DataFrame(currencies_new_data[1][1:], columns=currencies_new_data[1][0])
    currency1_df.set_index("time_start", inplace=True)
    currency2_df.set_index("time_start", inplace=True)

    return currency1_df, currency2_df


if __name__ == "__main__":
    RESEARCH_START_DATE = pd.Timestamp("2018-03-01")
    COINTEGRATION_RESEARCH_DATE = pd.Timestamp("2018-03-10")
    result = sample_pair_as_dollar_bars("candles/ETH-USD.csv", "candles/BTC-USD.csv", RESEARCH_START_DATE,
                                        COINTEGRATION_RESEARCH_DATE, 100000)