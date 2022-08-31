import numpy as np
from pandas import isna
from typing import List, Dict
from src.parsing.constants import ParameterData, ParameterType


class Normalization:
    def __init__(self, parameters_info: Dict[str, ParameterData]) -> None:
        """
            :param parameters_info: a dictionary mapping parameter name to an instance of ParameterData.
        """
        self.parameters_info = parameters_info
        self.encoding_dict = dict()

    def normalize(self, header: List[str], dataset: np.ndarray) -> np.ndarray:
        """
            :param header: a list of strings describing the header of the dataset
            :param dataset: a numpy array representing the dataset

            :return: a normalized version of `dataset`.
                     Only integer and real parameters are normalized using Min-Max normalization.
        """
        dataset = dataset.T
        new_values = []
        for idx, parameter_name in enumerate(header):
            if self.parameters_info[parameter_name].type != ParameterType.categorical:
                lb = self.parameters_info[parameter_name].values.lower_bound()
                ub = self.parameters_info[parameter_name].values.upper_bound()
                f = np.vectorize(lambda x: (float(x) - lb)/(ub - lb) if not isna(x) else x)
                c = f(dataset[idx, :].astype(np.float32))
                new_values.append(c)
            else:
                new_values.append(dataset[idx, :])
        return np.array(new_values).T


