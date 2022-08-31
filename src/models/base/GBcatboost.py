import numpy as np
from src.models.regression_model import RegressionModel
from src.parsing.constants import ParameterData, RegressionModelSetup, ParameterType
from catboost import CatBoostRegressor
from pandas import DataFrame, isna
from typing import Union, Dict


class GBcatboost(RegressionModel):
    def __init__(self, parameters_info: Dict[str, ParameterData], model_info: RegressionModelSetup) -> None:

        self.check_inputs(model_info.model_parameters, model_info.preprocessing)
        self.parameters_info = parameters_info
        self.preprocessing_strategy = model_info.preprocessing
        self.model = CatBoostRegressor(**model_info.model_parameters)
        self.encoding = None
        self.imputation = None
        self.header = []

    @staticmethod
    def check_inputs(model_parameters: Dict[str, Union[float, int, str]],
                     preprocessing: Dict[str, Union[float, int, str]]) -> None:
        """
            :param model_parameters: a dictionary containing the parameters to be passed to RangerForestRegressor
            :param preprocessing: a dictionary specifying the preprocessing options

            This method checks if all the required fields are present in the preprocessing and
            model_parameters dictionaries.
        """

        return

    def preprocess_data(self, X: DataFrame) -> DataFrame:
        # Sort dataframe columns based on parameter index
        X = X.reindex(sorted(X.columns, key=lambda x: self.parameters_info[x].index), axis=1)
        X = X.fillna("-")
        return X

    def fit(self, X: DataFrame, y: np.ndarray) -> None:
        X = self.preprocess_data(X)
        self.model.fit(X, y, cat_features=[par for par in X.columns if
                                           self.parameters_info[par].type == ParameterType.categorical])

    def predict(self, values: Dict[str, Union[str, int, float]]) -> float:
        x = np.array([t[1] for t in sorted(values.items(), key=lambda e: self.parameters_info[e[0]].index)], dtype=object)
        x[isna(x)] = "-"

        return self.model.predict(x)
