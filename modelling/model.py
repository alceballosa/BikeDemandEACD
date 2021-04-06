import typing as t

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

class BikeRentalFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_columns]


class BikeColumnTransformer(BaseEstimator, TransformerMixin):
    """Bike column transformer that processes categorical features
    for usage with XGBoost regressor.

    Since we are working only with tree algorithms, we do not do preprocessing on the
    continuous variables.
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )

    def fit(self, X, y=None):
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        X_ = self._column_transformer.transform(X)
        return X_


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "selector": BikeRentalFeatureSelection,
        "column_transformer": BikeColumnTransformer,
        "regressor": xgb.XGBRegressor,
    }