from __future__ import annotations
from abc import ABC, abstractmethod
from pandas import DataFrame
from numpy import ndarray
from src.parsing.constants import ParameterData, RegressionModelSetup
from typing import Union, Dict


class RegressionModel(ABC):

    @abstractmethod
    def __init__(self, parameters_info: Dict[str, ParameterData], model_info: RegressionModelSetup) -> None:
        pass

    @abstractmethod
    def fit(self, X: DataFrame, y: ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, values: Dict[str, Union[str, int, float]]) -> float:
        pass
