import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

import utils

PLOT_INTERVAL = 400
MEAN_WINDOW_SIZE = 60
SKIP_MIN = 60


def open_position(prices, assets, hedging_weights, is_buy, position, deal_prices):
    sign = 1 if is_buy else -1
    if np.isnan(prices).any():
        print("Can not trade now")
        return False
    else:
        for price, asset, hedge_weight in zip(prices, assets, hedging_weights):
            position[asset] = sign * hedge_weight * position["USD"] / price
            deal_prices[asset] = price
        position["USD"] = 0
        return True

def close_position(prices, assets, position, deal_prices):
    if np.isnan(prices).any():
        print("Can not trade now")
        return True
    else:
        for price, asset in zip(prices, assets):
            deal_price = deal_prices[asset]
            if position[asset] > 0:
                position["USD"] += price * position[asset]
            else:
                position["USD"] += -(deal_price + (deal_price - price)) * position[asset]
            position[asset] = 0
        return False

start_money = 1000
start_date = pd.Timestamp("2018-03-10")
data = utils.combine_market_data("candles/", "USD", "open", True,
                                 file_names_to_exclude=["ZRX-USD.csv", "ETC-USD.csv"],
                                 is_interpolate=False)

traded_assets = ['ETH-USD', 'BCH-USD', 'LTC-USD']
data = data[traded_assets]
position = {"USD": start_money}
deal_prices = dict()
for asset in traded_assets:
    position[asset] = 0
    deal_prices[asset] = 0

data = data.loc[start_date:]
assets_weighs = np.array([-0.00068367,  0.00123925, -0.00032639])
cointegrated_ts = data.copy() * assets_weighs
cointegrated_ts = cointegrated_ts[traded_assets[0]] + cointegrated_ts[traded_assets[1]] + cointegrated_ts[traded_assets[2]]

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

poses_usd = []
is_opened_position = False


def do_trading(i):
    global SKIP_MIN
    global poses_usd
    global is_opened_position

    i += SKIP_MIN
    current_minute_data = data.iloc[i]
    prices = current_minute_data.values
    cur_time = str(current_minute_data.name)

    interval_start = max(i - PLOT_INTERVAL, 0)
    interval_end = i + 1

    ax1.clear()
    selected_ts = cointegrated_ts[interval_start:interval_end]
    selected_ts.plot(ax=ax1, label="Cointegrating Combination")
    selected_ts.interpolate(method="linear", limit_direction="both", inplace=True)

    mean_prices = selected_ts.rolling(MEAN_WINDOW_SIZE).apply(lambda x: np.nanmean(x))
    mean_prices.plot(ax=ax1, label="Mean")
    std_prices = selected_ts.rolling(MEAN_WINDOW_SIZE).std()
    (mean_prices + std_prices).plot(ax=ax1, label="mean + 1 std", linestyle='dashed')
    (mean_prices - std_prices).plot(ax=ax1, label="mean - 1 std", linestyle='dashed')
    ax1.title.set_text(f"{traded_assets[0]}, {traded_assets[1]} stationary combination")
    ax1.legend()

    ax2.clear()
    usd_pos = position["USD"]
    for price, asset in zip(prices, data.columns):
        pos = position[asset]
        if pos > 0:
            usd_pos += price * pos
        else:
            deal_price = deal_prices[asset]
            usd_pos += -(deal_price + (deal_price - price)) * pos
    if np.isnan(usd_pos):
        usd_pos = poses_usd[-1]
    poses_usd.append(usd_pos)
    if len(poses_usd) > PLOT_INTERVAL:
        poses_usd = poses_usd[1:]

    ax2.plot(selected_ts.index[-len(poses_usd):], poses_usd)
    ax2.title.set_text("Portfolio value")

    ax3.clear()
    data[traded_assets[0]][interval_start:interval_end].plot(ax=ax3)
    ax3.title.set_text(traded_assets[0])

    ax4.clear()
    data[traded_assets[1]][interval_start:interval_end].plot(ax=ax4)
    ax4.title.set_text(traded_assets[1])

    plt.legend()

    if is_opened_position:
        if assets_weighs[1] > 0 and position[traded_assets[1]] > 0:
            cur_pos = "Long"
        else:
            cur_pos = "Short"
    else:
        cur_pos = "Flat"
    action = input(f"{cur_time} || {cur_pos} || choose action:")
    hedging_weights = prices * assets_weighs
    hedging_weights /= np.absolute(hedging_weights).sum()

    print(action, cur_pos, is_opened_position, hedging_weights, position, usd_pos)
    if action == "stop":
        sys.exit()
    elif action == "buy" and cur_pos != "Long":
        if is_opened_position:
            is_opened_position = close_position(prices, data.columns, position, deal_prices)
        else:
            is_opened_position = open_position(prices, data.columns, hedging_weights, True, position, deal_prices)
    elif action == "buy" and cur_pos == "Long":
        print("Not enough cash to buy more")
    elif action == "sell" and cur_pos != "Short":
        if is_opened_position:
            is_opened_position = close_position(prices, data.columns, position, deal_prices)
        else:
            is_opened_position = open_position(prices, data.columns, hedging_weights, False, position, deal_prices)
    elif action == "sell" and cur_pos == "Short":
        print("Not enough cash to sell more")
    else:
        try:
            skip = int(action)
        except ValueError:
            pass
        else:
            SKIP_MIN += skip


ani = animation.FuncAnimation(fig, do_trading, interval=3)
plt.show()