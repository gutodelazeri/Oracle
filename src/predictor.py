import pandas as pd
import numpy as np
import sys
import importlib
from src.scenario import Scenario
from parsing.constants import RegressionModelSetup, TrainingDataSetup, ParameterType, ConfigKeywords
from typing import Tuple, Dict


class Predictor:
    def __init__(self, scenario: Scenario, model_description: RegressionModelSetup, training_data: TrainingDataSetup) -> None:
        """
            :param scenario: information about each parameter
            :param model_description: information about the desired regression model
            :param training_data: information about the training data
        """
        self.scenario = scenario
        self.model = None
        X, y = self.construct_dataframe(training_data)
        self.construct_model(model_description, X, y)

    @staticmethod
    def get_model(path: str, name: str):
        module = importlib.import_module(path)
        my_class = getattr(module, name)
        return my_class

    def construct_dataframe(self, training_data: TrainingDataSetup) -> Tuple[pd.DataFrame, np.ndarray]:
        """
            :param training_data: information about the training data (e.g., file path, separator, name of the column
                                  containing the labels).

            :return: the design matrix X and the labels y. The design matrix X is a DataFrame of type 'object'
                     and the labels are represented as a 1D numpy array of floats.
        """
        X = pd.read_csv(training_data.dataset_path, sep=rf'{training_data.separator}', dtype=object)
        y = X[training_data.labels_column].to_numpy(dtype=float)
        X.drop(labels=training_data.labels_column, inplace=True, axis=1)

        # Check if scenario and dataset header are consistent
        scenario_params = set(self.scenario.get_parameters())
        dataset_params = set(X.columns)
        if scenario_params != dataset_params:
            sys.stderr.write(f"Some parameters defined in the scenario file do not match with the dataset header:\n")
            for par in scenario_params.union(dataset_params).difference(scenario_params.intersection(dataset_params)):
                sys.stderr.write(f"         {par}\n")
            exit(1)
        return X, y

    def construct_model(self, model_description: RegressionModelSetup, X: pd.DataFrame, y: np.ndarray) -> None:
        """
            :param model_description: information about the desired regression model (e.g., model name, parameters,
                                      preprocessing steps).
            :param X: design matrix
            :param y: labels

            This method initializes and trains the regression model.
        """
        try:
            model_constructor = self.get_model(model_description.model[ConfigKeywords.model_path], model_description.model[ConfigKeywords.model_name])
            self.model = model_constructor(self.scenario.get_parameters(), model_description)
            self.model.fit(X, y)
        except RuntimeError:
            sys.stderr.write(f"Error while loading regression model!\n")
            exit(1)

    def predict(self, par_values: Dict[str, str]) -> float:
        """
            :param par_values: a dictionary with all the parameter values (e.g., {"alpha": "0.1", "algorithm": "eas", "dlb": "0", "ants": "16", "rasrank": "", ...,})
            
            :return: a number representing the prediction for the configuration defined by par_values.
        """
        configuration = dict()

        for par_name, par_value in par_values.items():
            par_type = self.scenario.get_par_type(par_name)
            if par_value == "":
                configuration[par_name] = np.nan
            elif par_type == ParameterType.integer:
                configuration[par_name] = int(par_value)
            elif par_type == ParameterType.real:
                configuration[par_name] = float(par_value)
            else:
                configuration[par_name] = par_value

        prediction = self.model.predict(configuration)

        return prediction
