from collections import defaultdict

import numpy as np
import pytest

import utils


@pytest.fixture()
def default_pairs_coint_weights():
    coint_weights = {('ETH-USD', 'BTC-USD'): np.array([ 0.00285342, -0.00581943]),
                     ('ETH-USD', 'BCH-USD'): np.array([ 0.00309221, -0.00508119]),
                     ('ETH-USD', 'LTC-USD'): np.array([ 0.00307014, -0.00536644]),
                     ('BTC-USD', 'BCH-USD'): np.array([ 0.00073055, -0.00077677]),
                     ('BTC-USD', 'LTC-USD'): np.array([ 0.00120757, -0.00029359]),
                     ('BCH-USD', 'LTC-USD'): np.array([ 0.00261205, -0.00180342])}

    return coint_weights


@pytest.fixture()
def default_pairs_for_trading():
    return [('ETH-USD', 'BTC-USD'), ('ETH-USD', 'BCH-USD'), ('ETH-USD', 'LTC-USD')]


@pytest.fixture()
def default_pairs_z_scores():
    pairs_z_scores = {('ETH-USD', 'BTC-USD'): 2,
                      ('ETH-USD', 'BCH-USD'): 2,
                      ('ETH-USD', 'LTC-USD'): 2,
                      ('BTC-USD', 'BCH-USD'): 2,
                      ('BTC-USD', 'LTC-USD'): 2,
                      ('BCH-USD', 'LTC-USD'): 2}

    return pairs_z_scores


def test_select_pairs(default_pairs_coint_weights, default_pairs_z_scores):

    pairs_for_trading = utils.select_pairs(default_pairs_z_scores, default_pairs_coint_weights, 1)

    assert pairs_for_trading == [('ETH-USD', 'BTC-USD'), ('ETH-USD', 'BCH-USD'), ('ETH-USD', 'LTC-USD')]


def test_capital_weights(default_pairs_coint_weights, default_pairs_for_trading, default_pairs_z_scores):
    assets_weight, asset_in_pair_weights = utils.capital_weights(default_pairs_z_scores, default_pairs_for_trading,
                                                                 default_pairs_coint_weights)
    assets_weights_abs_sum = 0
    assets_weights_sum = 0
    for value in assets_weight.values():
        assets_weights_abs_sum += abs(value)
        assets_weights_sum += value

    assert assets_weights_abs_sum - 1 <= 0.001
    assert assets_weights_sum < 1

    asset_percent = defaultdict(float)
    for pair, assets_percents in asset_in_pair_weights.items():
        for asset_idx, asset in enumerate(pair):
            asset_percent[asset] += assets_percents[asset_idx]

    for asset, percent in asset_percent.items():
        assert abs(percent) == 1
