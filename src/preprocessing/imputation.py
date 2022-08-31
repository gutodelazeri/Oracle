import sys
import random as rd
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from src.parsing.constants import PreprocessingKeywords, ParameterType, ParameterData


class Imputation:

    def __init__(self, strategy: str, parameters_info: Dict[str, ParameterData], seed=1010101) -> None:
        """
            :param strategy: a string specifying the encoding strategy (e.g., "one hot")
            :param parameters_info: a dictionary mapping parameter name to an instance of ParameterData.
        """
        self.parameters_info = parameters_info
        self.strategy = strategy
        self.imputation_map: Dict[str, Union[float, int, str]] = dict()

        rd.seed(seed)

        self.create_imputation_map()

    def get_random_value(self, par_name: str) -> Union[float, int, str]:
        """
            :param par_name: a string representing a parameter name (e.g., "alpha")
            :return: a random value for par_name (e.g., 1.23)
        """
        if self.parameters_info[par_name].type == ParameterType.real:
            lb = self.parameters_info[par_name].values.lower_bound()
            ub = self.parameters_info[par_name].values.upper_bound()
            return rd.uniform(lb, ub)
        elif self.parameters_info[par_name].type == ParameterType.integer:
            lb = self.parameters_info[par_name].values.lower_bound()
            ub = self.parameters_info[par_name].values.upper_bound()
            return rd.randint(lb, ub)
        else:
            number_of_options = len(self.parameters_info[par_name].values)
            return self.parameters_info[par_name].values[rd.randint(0, number_of_options - 1)]

    def create_imputation_map(self) -> None:
        """
            :return: This method fills self.imputation_map with the imputation value for each parameter.
                     If the imputation strategy is PreprocessingKeywords.random_imputation, then nothing is done
        """
        if self.strategy == PreprocessingKeywords.random_imputation:
            return

        for par_name in self.parameters_info.keys():
            if self.strategy == PreprocessingKeywords.fixed_default_imputation:
                self.imputation_map[par_name] = self.parameters_info[par_name].default
            elif self.strategy == PreprocessingKeywords.fixed_random_imputation:
                self.imputation_map[par_name] = self.get_random_value(par_name)

    def impute(self, header: List[str], dataset: np.ndarray) -> np.ndarray:
        """
            :param header: a list of strings describing the header of the dataset
            :param dataset: a numpy array representing the dataset
            :return: a pandas DataFrame where NA values were imputed according to the imputation strategy

            When the imputation is strategy is fixed_default_imputation or random_default_imputation, this
            method also fills self.imputation_dict with the imputation value for each parameter.
        """
        dataset = dataset.T
        if self.strategy == PreprocessingKeywords.fixed_random_imputation or self.strategy == PreprocessingKeywords.fixed_default_imputation:
            for idx, par_name in enumerate(header):
                mask = np.where(pd.isna(dataset[idx, :]))
                dataset[idx, mask] = self.imputation_map[par_name]
        elif self.strategy == PreprocessingKeywords.random_imputation:
            for idx, par_name in enumerate(header):
                mask = np.where(pd.isna(dataset[idx, :]))
                dataset[idx, mask] = np.array([self.get_random_value(par_name) for _ in range(mask[0].shape[0])])
        else:
            sys.stderr.write(
                f"Catastrophic bug at Imputation class: strategy named '{self.strategy}' was not recognized :(\n")
            exit(1)

        return dataset.T
