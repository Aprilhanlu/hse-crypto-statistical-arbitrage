# Recomender System for Cryptocurrencies Arbitrage

This repository contains a demo version of a recomender system for cryptocurrencies arbitrage. Recomender system is based on plain Pair Trading strategy.

`Research Pair Trading Opportunities.ipynb` is a brief overview of research devoted to searching cointegrated sets of stock. 

`demo.py` contains demo version of the project which allows you to trade ('ETH-USD', 'BCH-USD', 'LTC-USD') set of currencies on historical period from "2018-03-10" to "2018-12-14" using recomendation from recomender system. 

# Demo User Manual 

1. Clone the repository

2. Type in terminal `python3 -m virtualenv env`

3. Type in terminal `source env/bin/activate`

4. Type in terminal `pip3 install --upgrade pip`

5. Type in terminal `pip3 install -r requirements.txt`

6. Type in terminal `python3 demo.py`

You will see something like this:

![Alt text](media/rec_system_demo.png)
