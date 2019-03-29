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

You control the demo in your terminal. Every trading minute you will see the following lines:

`Your current position: {'USD': 1000, 'ETH-USD': 0, 'BCH-USD': 0, 'LTC-USD': 0}`
`2018-03-11 00:48:00 || Flat || choose action:`

`Your current position` shows how many `USD`, `ETH`, `BCH` or `LTC` you have. If you have a negative position in some of those assets, it means that you are shorting this asset and profit from its price decrease otherwise positive sign shows that you are long on this asset and you profit from a price increase. 

On the next line you see `2018-03-11 00:48:00` it is current date and minute in history.

`Flat` - means that you have no cryptocurrencies in your portfolio and that you stay entirely in `USD`. Also, this value could have the following values: `Long` means that now you are betting on the increase of the top left graph and `Short` means that now you are betting on the decrease of the graph. 

`choose action` Asks you to do one of the following action:

1. Press Enter - just skip the current minute.

2. Type `buy` - bet on the increase if we are flat or close out short position if we are short.

3. Type `sell` - bet on the decrease of the price if we are flat or close out long position if we are long

4. Type `int_number` - skip `int_number` minutes

5. Type `stop` to close demo




