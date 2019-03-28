import argparse
import datetime
import pickle
import os
import requests
import time
from copy import deepcopy

import dateparser
import numpy as np

PAIRS = [
"BTC-EUR",
"BCH-GBP",
"MKR-USDC",
"BCH-EUR",
"BTC-USD",
"ZEC-USDC",
"DNT-USDC",
"LOOM-USDC",
"DAI-USDC",
"GNT-USDC",
"ZIL-USDC",
"MANA-USDC",
"CVC-USDC",
"ETH-USDC",
"ZRX-EUR",
"BAT-USDC",
"ETC-EUR",
"BTC-USDC",
"ZRX-USD",
"ETH-BTC",
"ETH-EUR",
"ETH-USD",
"LTC-BTC",
"LTC-EUR",
"LTC-USD",
"ETC-USD",
"ETC-BTC",
"ZRX-BTC",
"ETC-GBP",
"ETH-GBP",
"LTC-GBP"
]
PATH = "/Users/nialeksandrov/University/diplom/"


def get_data(kind_of_data, pair, granularity, start_date, end_date):
    url_base = "https://api.pro.coinbase.com/products/"
    data = requests.get(url_base + pair + "/" + kind_of_data + "?granularity=" + str(granularity) + \
                        "&start=" + start_date + "&end=" + end_date)

    return data.json()


parser = argparse.ArgumentParser()
parser.add_argument("start_date", metavar="start_date", type=str)
parser.add_argument("final_date", metavar="final_date", type=str)
args = parser.parse_args()
start_date = dateparser.parse(args.start_date)
final_date = dateparser.parse(args.final_date)


five_hours_delta = datetime.timedelta(minutes=300)
one_minute_delta = datetime.timedelta(minutes=1)
for product_name in PAIRS:
    print(product_name)
    files_in_dir = os.listdir(PATH)
    if product_name + ".csv" in files_in_dir:
        print(product_name + " is already parsed")
        continue
    if product_name + ".pkl" in files_in_dir:
        trades_list = pickle.load(open(product_name + ".pkl", "rb"))
        begin = dateparser.parse(str(trades_list[-1])) + five_hours_delta + one_minute_delta
        trades_list = trades_list[:-1]
    else:
        trades_list = []
        begin = deepcopy(start_date)

    counter = 1
    while begin <= final_date:
        if counter % 100 == 0:
            pickle.dump(trades_list + [begin - five_hours_delta - one_minute_delta], open(product_name + ".pkl", "wb"))
            print("Array dumped")

        print(begin.isoformat())

        time.sleep(0.25)
        print(begin)
        end = begin + five_hours_delta
        try:
            data = get_data("candles", product_name, 60, begin.isoformat(), end.isoformat())
            if "message" in data:
                time.sleep(2)
                print(data)
            else:
                trades_list.extend(data[::-1])
                begin = end + one_minute_delta
        except Exception as e:
            print(e)
            print("Type something to continue")
            a = input()
            print("Continuing...")

        counter += 1

    trades_list = np.array(trades_list, dtype=np.float32)
    np.savetxt(product_name + ".csv", trades_list, delimiter=",", header="time, low, high, open, close, volume")




