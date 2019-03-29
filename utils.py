import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import zipline
from collections import defaultdict
from tqdm import tqdm_notebook
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
    for pair in market:
        pair_data = pd.read_csv(f"{folder}{pair}", index_col="time", parse_dates=["time"])
        serieses.append(pair_data[column])

    ticker_names = file_names_to_tickers(market)
    data = pd.concat(serieses, axis=1, join="outer")
    data.columns = ticker_names

    if is_interpolate:
        data = data.interpolate()

    if is_cut_nas:
        data.dropna(inplace=True)
    data = data.resample("1min").mean()
    return data


def create_zipline_panel(files_path, pairs, min_order_size):
    if not pairs[0].endswith(".csv"):
        pairs = map(lambda x: x + ".csv", pairs)
    data = dict()
    for pair in pairs:
        pair_data = pd.read_csv(f"{files_path}{pair}", index_col="time", parse_dates=["time"])
        pair_data.index.name = "date"
        pair_data = pair_data.resample("1min").mean()
        pair_data.loc[:, "volume"] /= min_order_size
        pair_data.loc[:, ["open", "high", "low", "close"]] *= min_order_size
        pair_name = pair[:-4]
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


def estimate_pairs_z_score(data, pairs_coint_weights, pairs_to_calc, window_size):
    pairs_z_scores = dict()
    for pair in pairs_to_calc:
        weights = pairs_coint_weights[pair]
        pair_sids = list(map(lambda x: zipline.api.symbol(x), pair))
        pair_values = data.history(pair_sids, fields="price", bar_count=window_size,
                                   frequency="1m")
        print(pair_values.iloc[0])
        pair_values = (pair_values * weights).sum(axis=1)
        pair_values.interpolate(limit_direction="both", inplace=True)
        z_score = (pair_values[-1] - pair_values.mean()) / pair_values.std()
        pairs_z_scores[pair] = z_score

    return pairs_z_scores


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


def adjust_positions_by_target_percent(context, data):
    for stock, target_percent in context.traded_stocks_target_percent.items():
        if target_percent != 0:
            zipline.api.order_target_percent(stock, target_percent)


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