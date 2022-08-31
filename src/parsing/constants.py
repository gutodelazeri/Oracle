from dataclasses import dataclass
from typing import Union, NewType, Tuple, List, Dict


class Range:
    def __init__(self, interval: Union[Tuple[int, int], Tuple[float, float]]) -> None:
        self.interval = interval

    def lower_bound(self) -> Union[int, float]:
        return self.interval[0]

    def upper_bound(self) -> Union[int, float]:
        return self.interval[1]


Listing = NewType('Listing', List[str])


@dataclass
class ParameterType:
    real = "r"
    integer = "i"
    categorical = "c"


@dataclass
class ScenarioKeywords:
    switch = "switch"
    type = "type"
    values = "values"
    default = "default"


@dataclass
class ConfigKeywords:
    SERVER_SETUP = "general"
    scenario_file = "scenario_file"
    port_number = "port"
    REGRESSION_MODEL = "model"
    model_path = "model_path"
    model_name = "model_name"
    MODEL_PARAMETERS = "model_parameters"
    PREPROCESSING = "preprocessing"
    preprocessing = "preprocessing"
    imputation = "imputation"
    encoding = "encoding"
    normalization = "normalization"
    TRAINING_DATA = "data"
    dataset_path = "dataset_path"
    separator = "separator"
    labels_column = "labels_column"


@dataclass
class PreprocessingKeywords:
    one_hot_encoding = "one-hot"
    binary_encoding = "binary"
    integer_encoding = "integer"

    fixed_random_imputation = "fixed random"
    fixed_default_imputation = "fixed default"
    random_imputation = "random"


@dataclass
class ParameterData:
    name: str
    switch: str
    type: ParameterType
    index: int
    values: Union[Range, Listing]
    default: Union[int, float, str]


@dataclass
class ServerSetup:
    scenario_file: str
    port_number: int


@dataclass
class RegressionModelSetup:
    model: Dict[str, Union[str, int, float]]
    model_parameters: Dict[str, Union[str, int, float]]
    preprocessing: Dict[str, Union[str, int, float]]


@dataclass
class TrainingDataSetup:
    dataset_path: str   # Path to the dataset file
    separator: str      # Character denoting the separation of values
    labels_column: str  # Name of the column containing the labels










