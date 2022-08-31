import sys
import numpy as np
from src.models.regression_model import RegressionModel
from src.parsing.constants import ConfigKeywords, ParameterData, RegressionModelSetup, ParameterType
from src.preprocessing.imputation import Imputation
from src.preprocessing.encoding import Encoding
from skranger.ensemble import RangerForestRegressor
from pandas import DataFrame
from typing import Union, Dict


class RFranger(RegressionModel):
    def __init__(self, parameters_info: Dict[str, ParameterData], model_info: RegressionModelSetup) -> None:

        self.check_inputs(model_info.model_parameters, model_info.preprocessing)
        self.parameters_info = parameters_info
        self.preprocessing_strategy = model_info.preprocessing
        self.model = RangerForestRegressor(**model_info.model_parameters)
        self.encoding = None
        self.imputation = None
        self.header = []
        self.categorical_features = []

    @staticmethod
    def check_inputs(model_parameters: Dict[str, Union[float, int, str]],
                     preprocessing: Dict[str, Union[float, int, str]]) -> None:
        """
            :param model_parameters: a dictionary containing the parameters to be passed to RangerForestRegressor
            :param preprocessing: a dictionary specifying the preprocessing options

            This method checks if all the required fields are present in the preprocessing and
            model_parameters dictionaries.
        """

        missing_fields = []
        if ConfigKeywords.imputation not in preprocessing:
            missing_fields.append(ConfigKeywords.imputation)
        if ConfigKeywords.encoding not in preprocessing:
            missing_fields.append(ConfigKeywords.encoding)

        if len(missing_fields) > 0:
            sys.stderr.write(f"You must specify the following fields in the {ConfigKeywords.PREPROCESSING} section:\n")
            for field in missing_fields:
                sys.stderr.write(f"     {field}\n")
            exit(1)

    def preprocess_data(self, X: DataFrame) -> np.ndarray:
        self.imputation = Imputation(self.preprocessing_strategy[ConfigKeywords.imputation], self.parameters_info)
        self.encoding = Encoding(self.preprocessing_strategy[ConfigKeywords.encoding], self.parameters_info)

        # Sort dataframe columns based on parameter index
        X = X.reindex(sorted(X.columns, key=lambda x: self.parameters_info[x].index), axis=1)
        self.header = list(X.columns)

        X = self.imputation.impute(self.header, X.to_numpy(dtype=object))
        X = self.encoding.encode(self.header, X)

        col = 0
        for p in self.header:
            if self.parameters_info[p].type == ParameterType.categorical:
                # Number of bits that are used to represent encoded values of the parameter p
                bits = len(list(self.encoding.encoding_dict[p].values())[0])
                self.categorical_features += [i for i in range(col, col+bits)]
                col += bits
            else:
                col += 1

        return X

    def fit(self, X: DataFrame, y: np.ndarray) -> None:
        X = self.preprocess_data(X)
        self.model.fit(X, y, categorical_features=self.categorical_features)

    def predict(self, values: Dict[str, Union[str, int, float]]) -> float:
        x = [t[1] for t in sorted(values.items(), key=lambda e: self.parameters_info[e[0]].index)]
        x = self.imputation.impute(self.header, np.array([x], dtype=object))
        x = self.encoding.encode(self.header, x)

        return self.model.predict(x)[0]
