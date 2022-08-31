import sys
import numpy as np
from typing import List, Union, Dict
from math import log2, ceil
from src.parsing.constants import ParameterData, PreprocessingKeywords, ParameterType


class Encoding:
    def __init__(self, strategy: str, parameters_info: Dict[str, ParameterData]) -> None:
        """
            :param strategy: a string specifying the encoding strategy (e.g., "one hot")
            :param parameters_info: a dictionary mapping parameter name to an instance of ParameterData.
        """
        self.parameters_info = parameters_info
        self.strategy = strategy
        self.encoding_dict = dict()

        self.create_encoding()

    @staticmethod
    def get_one_hot_representation(options: List[str]) -> Dict[Union[str, int], List[int]]:
        dimension = len(options)
        mapping = {op: [1 if idx == op_id else 0 for op_id in range(dimension)] for idx, op in enumerate(options)}
        mapping[np.nan] = [np.nan for _ in range(dimension)]
        return mapping

    @staticmethod
    def get_binary_representation(options: List[str]) -> Dict[Union[str, int], List[int]]:
        dimension = max(ceil(log2(len(options))), 1)
        mapping = {op: [int(i) for i in format(idx, f'0{dimension}b')] for idx, op in enumerate(options)}
        mapping[np.nan] = [np.nan for _ in range(dimension)]
        return mapping

    @staticmethod
    def get_integer_representation(options: List[str]) -> Dict[Union[str, int], List[int]]:
        mapping = {op: [idx] for idx, op in enumerate(options)}
        mapping[np.nan] = [np.nan]
        return mapping

    def create_encoding(self):
        """
            Create the rules for encoding each categorical parameter in self.parameters_info.
        """
        for name in self.parameters_info.keys():
            if self.parameters_info[name].type != ParameterType.categorical:
                continue
            options = self.parameters_info[name].values
            if self.strategy == PreprocessingKeywords.one_hot_encoding:
                self.encoding_dict[name] = self.get_one_hot_representation(options)
            elif self.strategy == PreprocessingKeywords.binary_encoding:
                self.encoding_dict[name] = self.get_binary_representation(options)
            elif self.strategy == PreprocessingKeywords.integer_encoding:
                self.encoding_dict[name] = self.get_integer_representation(options)
            else:
                sys.stderr.write(
                    f"Catastrophic bug at Encoding class: strategy named '{self.strategy}' was not recognized :(\n")
                exit(1)

    def encode_array(self, name: str, values: np.ndarray) -> np.ndarray:
        """
            :param name: a string specifying a parameter name (e.g., "alpha")
            :param values: a Numpy Array with values for the parameter specified in the first argument.

            :return: a Numpy array A where A[i] is the encoding of values[i].
                    For example, if the encoding strategy is "binary", 'name' is "localsearch", and 'values' is Array(["1", "2", "1", "3"]),
                    then the following mapping would be defined
                            "0" -> [0, 0]
                            "1" -> [0, 1]
                            "2" -> [1, 0]
                            "3" -> [1, 1]
                    and the output would be Array([[0, 1], [1, 0], [0, 1], [1, 1]])
        """

        # Encode values
        new_values = np.array(list(map(lambda x: self.encoding_dict[name][x], values)))

        return new_values

    def encode(self, header: List[str], dataset: np.ndarray) -> np.ndarray:
        """
            :param header: a list of strings describing the header of the dataset
            :param dataset: a numpy array containing the dataset

            :return: the encoding of the parameter 'name', as defined by the encoding
                    strategy 'self.strategy'
        """
        new_values = []
        dataset = dataset.T
        for idx, parameter_name in enumerate(header):
            if self.parameters_info[parameter_name].type != ParameterType.categorical:
                column = dataset[idx, :]
                column = column.reshape(column.shape[0], 1)
                new_values.append(column)
                continue

            # Encode parameter_name
            encoded_values = self.encode_array(parameter_name, dataset[idx, :])

            new_values.append(encoded_values)

        new_dataset = np.concatenate(new_values, axis=1)

        return new_dataset
