# Recomender System for Cryptocurrencies Arbitrage

This repository contains a demo version of a recommender system for cryptocurrencies arbitrage. The recommender system is based on plain Pair Trading strategy.

`Research Pair Trading Opportunities.ipynb` is a brief overview of research devoted to searching cointegrated sets of stock. 

`demo.py` contains demo version of the project which allows you to trade ('ETH-USD', 'BCH-USD', 'LTC-USD') set of currencies on a historical period from "2018-03-10" to "2018-12-14" using recommendations from the recommender system. 

# Demo User Manual 

1. Clone the repository

2. Type in terminal `python3 -m virtualenv env`

3. Type in terminal `source env/bin/activate`

4. Type in terminal `pip3 install --upgrade pip`

5. Type in terminal `pip3 install -r requirements.txt`

6. Type in terminal `python3 demo.py`

You will see something like this:

![Alt text](media/rec_system_demo.png)

Here on the screenshot, you trade ETH-USD and BTH-USD cointegrating set, its price is showed on the top left graph as the blue line. When current price is above the green dashed line it is a signal to short or sell, when it is bellow it is a signal to buy. On the top right graph, your current portfolio value is shown. On the graphs below ETH-USD and BCH-USD prices are shown. 
