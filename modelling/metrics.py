import numpy as np
from sklearn.metrics import make_scorer


def bike_demand_error(y_true, y_pred, understock_price=0.5, overstock_price=0.5):
    """
    Bike demand error which allows us to differently weight overestimation
    and underestimation of bike demand.
    """
    error = (y_true - y_pred).astype(np.float32)
    factor = np.zeros_like(error)
    factor[error > 0] = understock_price
    factor[error < 0] = overstock_price
    return np.mean((np.abs(error) * factor))


def get_metric_name_mapping():
    return {_bde(): bike_demand_error}


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _bde(): make_scorer(bike_demand_error, greater_is_better=False, **params)
    }
    return mapping[name]


def _bde():
    return "bike demand error"
