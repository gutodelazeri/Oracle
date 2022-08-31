import sys
import numpy as np
from src.models.regression_model import RegressionModel
from src.parsing.constants import ParameterData, RegressionModelSetup, ConfigKeywords, ParameterType
from src.preprocessing.normalization import Normalization
from src.preprocessing.encoding import Encoding
from sklearn.neighbors import BallTree
from pandas import DataFrame, isna
from typing import Union, List, Tuple, Dict
from math import sqrt


class WeightedInterpolation(RegressionModel):
    def __init__(self, parameters_info: Dict[str, ParameterData], model_info: RegressionModelSetup) -> None:
        self.check_inputs(model_info.model_parameters, model_info.preprocessing)
        self.parameters_info = parameters_info
        self.preprocessing_strategy = model_info.preprocessing
        self.model_parameters = model_info.model_parameters
        self.tree = None
        self.X = None
        self.y = None
        self.normalization = None
        self.encoding = None
        self.header = None
        self.NA = -1

        self.check_inputs(self.model_parameters, model_info.preprocessing)

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

        if "p" not in model_parameters:
            missing_fields.append("p")
        if "k" not in model_parameters:
            missing_fields.append("k")

        if len(missing_fields) > 0:
            sys.stderr.write(
                f"You must specify the following fields in the {ConfigKeywords.MODEL_PARAMETERS} section:\n")
            for field in missing_fields:
                sys.stderr.write(f"     {field}\n")
            exit(1)

        return

    @staticmethod
    def distance_metric(v1, v2):
        """
            :param v1: a list of numbers with n elements
            :param v2: a list of numbers with n elements
            :return: a float

            It computes the distance between v1 and v2 according to the Heterogeneous Euclidean-Overlap Metric.

            (see Wilson, D. R., & Martinez, T. R. (1997). Improved heterogeneous distance functions. Journal of Artificial
            Intelligence Research, 6, 1-34.)
        """
        acc = 0
        NA = WeightedInterpolation.distance_metric.NA
        for parameter_index, (x, y) in enumerate(zip(v1, v2)):
            if x == NA or y == NA:
                acc += 1
            elif parameter_index in WeightedInterpolation.distance_metric.categorical_parameters:
                acc += int(x != y)
            else:
                acc += pow(x - y, 2)

        return sqrt(acc)

    def get_neighbors(self, x: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
            :param x: : a 1D numpy array
            :return: A list of tuples containing the k nearest neighbors of x, together with their values

        """
        ind = self.tree.query(x, k=self.model_parameters["k"], return_distance=False)[0]

        return [(self.X[i, :], self.y[i]) for i in ind]

    def inverse_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
            :param x1: a numpy 1D array
            :param x2: a numpy 1D array
            :return: a float

            It returns the Inverse distance weight (IDW)
        """

        # Euclidean distance
        d = np.linalg.norm(x1 - x2)
        if d < 0.001:
            return 0.0
        else:
            return 1.0 / pow(d, self.model_parameters["p"])

    def interpolate(self, x: np.ndarray, neighbors: List[Tuple[np.ndarray, float]]) -> float:
        """
            :param x: a 1D numpy array
            :param neighbors: a list of tuples, where the first component of each tuple is
                              a numpy array (the coordinate) and the second component is its value

            :return: a float

            It returns the interpolation value, according to Shepard's method
            (refer to https://en.wikipedia.org/wiki/Inverse_distance_weighting)
        """
        weights = []
        for neighbor in neighbors:
            coordinate = neighbor[0]
            value = neighbor[1]
            acc = self.inverse_distance(x, coordinate)
            if acc == 0:
                return value
            else:
                weights.append(acc)

        weighted_sum = 0
        for idx, neighbor in enumerate(neighbors):
            weighted_sum += neighbor[1] * weights[idx]

        return weighted_sum / sum(weights)

    def preprocess_data(self, X: DataFrame) -> np.ndarray:

        self.normalization = Normalization(self.parameters_info)
        self.encoding = Encoding("integer", self.parameters_info)

        X = X.reindex(sorted(X.columns, key=lambda x: self.parameters_info[x].index), axis=1)
        self.header = list(X.columns)

        X = self.normalization.normalize(self.header, X.to_numpy())
        X = self.encoding.encode(self.header, X)

        # Replaces NAN by -1
        X[isna(X)] = self.NA

        return X

    def fit(self, X: DataFrame, y: np.ndarray) -> None:
        WeightedInterpolation.distance_metric.categorical_parameters = [info.index for info in
                                                                        self.parameters_info.values()]
        WeightedInterpolation.distance_metric.NA = self.NA
        X = self.preprocess_data(X)
        self.tree = BallTree(X, leaf_size=4, metric='pyfunc', func=WeightedInterpolation.distance_metric)
        self.X = X
        self.y = y

    def predict(self, values: Dict[str, Union[str, int, float]]) -> float:

        x = [t[1] for t in sorted(values.items(), key=lambda e: self.parameters_info[e[0]].index)]
        x = self.normalization.normalize(self.header, np.array([x], dtype=object))
        x = self.encoding.encode(self.header, x)

        x = x.astype(np.float32)
        x[isna(x)] = self.NA

        neighbors = self.get_neighbors(x)
        prediction = self.interpolate(x, neighbors)

        return prediction
