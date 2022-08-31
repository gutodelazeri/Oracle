import sys
import toml
from src.parsing.constants import *


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def check_par_info(par_name: str, par_info: Dict[str, Union[str, int, float]]) -> None:
        """
            :param par_name: a parameter name (e.g. "beta")
            :param par_info: a dictionary containing information about the parameter par_name
            This method checks if par_info contains all the required information about the parameter par_name.
            The required fields of par_info are:
                "switch", the switch used in queries to the Oracle (e.g. "--beta")
                "type", the type identifier (e.g. "r")
                "values", a list defining the range of possible values (e.g. [0.00, 10.00])
                "default", the default value of this parameter (e.g. 1.2)
        """

        if type(par_name) is not str:
            sys.stderr.write(f"The parameter name of {par_name} should be a string!\n")
            exit(1)

        missing_info = []
        for info in (ScenarioKeywords.switch, ScenarioKeywords.type, ScenarioKeywords.values, ScenarioKeywords.default):
            if info not in par_info:
                missing_info.append(info)
        if len(missing_info) != 0:
            sys.stderr.write(f"You must specify the following fields for parameter {par_name}:\n")
            for field in missing_info:
                sys.stderr.write(f"     {field}\n")
            exit(1)

    @staticmethod
    def is_a_valid_type(par_name: str, type_identifier: str) -> None:
        """
            :param par_name: a parameter name (e.g. "alpha")
            :param type_identifier: the type identifier (e.g. "r")

            This method checks if type_identifier is a valid type identifier.
            Valid type identifiers are (February 8th, 2022):
                "c", for categorical parameters
                "i", for integer parameters
                "r", for real parameters
        """

        if type_identifier not in (ParameterType.real, ParameterType.integer, ParameterType.categorical):
            sys.stderr.write(f"Unknown parameter type identifier for parameter {par_name}: {type_identifier}\n")
            exit(1)

    @staticmethod
    def is_a_valid_range(par_name: str, type_identifier: str, values: Union[Range, Listing]) -> None:
        """
            :param par_name: a parameter name (e.g. "algorithm")
            :param type_identifier: the type identifier (e.g. "c")
            :param values: an instance of Range or an instance of Listing

            This method checks if values is a valid Range or a valid Listing for the parameter par_name.
        """
        if type_identifier in (ParameterType.real, ParameterType.integer):
            if len(values) != 2 or values[0] >= values[1]:
                sys.stderr.write(f"Parameter {par_name} was given an invalid range! {values}\n")
                exit(1)
        elif len(values) == 0:
            sys.stderr.write(f"Categorical parameter {par_name} needs at least one valid value!\n")
            exit(1)

    @staticmethod
    def is_a_valid_default(par_name: str, type_identifier: str, values: Union[Range, Listing],
                           default_value: Union[str, int, float]) -> None:
        """
            :param par_name: a parameter name (e.g. "nnls")
            :param type_identifier: the type identifier (e.g. "i")
            :param values: an instance of Range or an instance of Listing defining the range of possible values for the given parameter
            :param default_value: the default value of parameter par_name (e.g. 10)

            This method checks if default_value is a valid default value for par_name
        """
        if (type_identifier == ParameterType.categorical) and (type(default_value) is not str):
            sys.stderr.write(f"Default value for parameter {par_name} should be a string!\n")
            exit(1)
        elif (type_identifier == ParameterType.integer) and (type(default_value) is not int):
            sys.stderr.write(f"Default value for parameter {par_name} should be an integer number!\n")
            exit(1)
        elif (type_identifier == ParameterType.real) and (type(default_value) is not float) and (
                type(default_value) is not int):
            sys.stderr.write(f"Default value for parameter {par_name} should be a real number!\n")
            exit(1)
        else:
            if (type_identifier == ParameterType.categorical) and (default_value not in values):
                sys.stderr.write(f"Default value for parameter {par_name} was not listed in the field 'values'!\n")
                exit(1)
            if (type_identifier != ParameterType.categorical) and (not (values[0] <= default_value <= values[1])):
                sys.stderr.write(f"Default value for parameter {par_name} is out of range!\n")
                exit(1)

    @staticmethod
    def check_configuration(config_data: Dict[str, Union[str, int, float]]) -> None:
        """
            config_data: a dictionary containing all the information about the server setup

            This method checks if config_data contains the minimum necessary information
        """
        missing_fields = []

        if ConfigKeywords.SERVER_SETUP not in config_data:
            missing_fields.append(ConfigKeywords.SERVER_SETUP)
        else:
            if ConfigKeywords.scenario_file not in config_data[ConfigKeywords.SERVER_SETUP]:
                missing_fields.append(f'{ConfigKeywords.SERVER_SETUP}.{ConfigKeywords.scenario_file}')
            if ConfigKeywords.port_number not in config_data[ConfigKeywords.SERVER_SETUP]:
                missing_fields.append(f'{ConfigKeywords.SERVER_SETUP}.{ConfigKeywords.port_number}')

        if ConfigKeywords.REGRESSION_MODEL not in config_data:
            missing_fields.append(ConfigKeywords.REGRESSION_MODEL)
        else:
            if ConfigKeywords.model_name not in config_data[ConfigKeywords.REGRESSION_MODEL]:
                missing_fields.append(f'{ConfigKeywords.REGRESSION_MODEL}.{ConfigKeywords.model_name}')
            if ConfigKeywords.model_path not in config_data[ConfigKeywords.REGRESSION_MODEL]:
                missing_fields.append(f'{ConfigKeywords.REGRESSION_MODEL}.{ConfigKeywords.model_path}')

        if ConfigKeywords.MODEL_PARAMETERS not in config_data:
            missing_fields.append(ConfigKeywords.MODEL_PARAMETERS)

        if ConfigKeywords.PREPROCESSING not in config_data:
            missing_fields.append(ConfigKeywords.PREPROCESSING)

        if ConfigKeywords.TRAINING_DATA not in config_data:
            missing_fields.append(ConfigKeywords.TRAINING_DATA)
        else:
            if ConfigKeywords.dataset_path not in config_data[ConfigKeywords.TRAINING_DATA]:
                missing_fields.append(f'{ConfigKeywords.TRAINING_DATA}.{ConfigKeywords.dataset_path}')
            if ConfigKeywords.separator not in config_data[ConfigKeywords.TRAINING_DATA]:
                missing_fields.append(f'{ConfigKeywords.TRAINING_DATA}.{ConfigKeywords.separator}')
            if ConfigKeywords.labels_column not in config_data[ConfigKeywords.TRAINING_DATA]:
                missing_fields.append(f'{ConfigKeywords.TRAINING_DATA}.{ConfigKeywords.labels_column}')

        if len(missing_fields) > 0:
            sys.stderr.write("The following fields are missing:\n")
            for field in missing_fields:
                sys.stderr.write(f"     > {field}\n")
            exit(1)

    def parse_scenario_file(self, file_name: str) -> Dict[str, ParameterData]:
        """
            :param file_name: string representing a path to a .toml file

            :return: A dictionary mapping a parameter name to an instance of ParameterData.

            This method opens and parses the file file_name, which is supposed to be a parameter description file.
        """
        par_data = toml.load(file_name)

        parameters = dict()
        for index, parameter in enumerate(sorted(par_data.keys())):
            parameter_fields = par_data[parameter]

            # It's important that the checks are executed in this order
            self.check_par_info(parameter, parameter_fields)
            self.is_a_valid_type(parameter, parameter_fields[ScenarioKeywords.type])
            self.is_a_valid_range(parameter, parameter_fields[ScenarioKeywords.type],
                                  parameter_fields[ScenarioKeywords.values])
            self.is_a_valid_default(parameter, parameter_fields[ScenarioKeywords.type],
                                    parameter_fields[ScenarioKeywords.values], parameter_fields[ScenarioKeywords.default])

            if parameter_fields[ScenarioKeywords.type] != ParameterType.categorical:
                vals = Range(parameter_fields[ScenarioKeywords.values])
            else:
                vals = parameter_fields[ScenarioKeywords.values]

            new_parameter = ParameterData(name=parameter,
                                          switch=parameter_fields[ScenarioKeywords.switch],
                                          type=parameter_fields[ScenarioKeywords.type],
                                          values=vals,
                                          default=parameter_fields[ScenarioKeywords.default],
                                          index=index)

            parameters[parameter] = new_parameter

        return parameters

    def parse_config_file(self, file_name: str) -> Tuple[ServerSetup, RegressionModelSetup, TrainingDataSetup]:
        """
            :param file_name: a string representing a path to a .toml file

            :return: a tuple (a, b, c), where 'a' is an instance of ServerSetup, 'b is an instance of RegressionModel, and
                     'c' is an instance of TrainingData

            This method parses a .toml configuration file.
        """
        config_data = toml.load(file_name)

        self.check_configuration(config_data)

        server = ServerSetup(scenario_file=config_data[ConfigKeywords.SERVER_SETUP][ConfigKeywords.scenario_file],
                             port_number=config_data[ConfigKeywords.SERVER_SETUP][ConfigKeywords.port_number])

        model = RegressionModelSetup(model=config_data[ConfigKeywords.REGRESSION_MODEL],
                                     model_parameters=config_data[ConfigKeywords.MODEL_PARAMETERS],
                                     preprocessing=config_data[ConfigKeywords.PREPROCESSING])

        training_data = TrainingDataSetup(
            dataset_path=config_data[ConfigKeywords.TRAINING_DATA][ConfigKeywords.dataset_path],
            separator=config_data[ConfigKeywords.TRAINING_DATA][ConfigKeywords.separator],
            labels_column=config_data[ConfigKeywords.TRAINING_DATA][ConfigKeywords.labels_column])

        return server, model, training_data

    @staticmethod
    def parse_command_line_argument(command: str, switches: List[str]) -> Dict[str, str]:
        """
            :param command: a string representing a command line argument (e.g., "--alpha 1.2 --algorithm eas ...")
            :param switches: a list of switches to look for in the input string (e.g., ["--alpha", "--algorithm", ...])

            :return: a dictionary mapping a parameter switch to the value found in the input string. If a switch in the list
                    of switches is not present in the command line argument, it is mapped to the empty string.

                    Example:
                        command = "--alpha 1.2 --beta 5.34 --rho 0.531"
                        switches = ["--alpha", "--beta", "--rho", "--algorithm"]

                        return: {"--alpha": "1.2", "--beta": "5.34", "rho": "0.531", "--algorithm": ""}
        """
        par_values = dict()
        command_length = len(command)
        for switch in switches:
            switch_length = len(switch)
            switch_start_index = command.find(switch)

            if switch_start_index != -1:
                value_start_index = switch_start_index + switch_length
                value_end_index = command[value_start_index:].find(" ") % command_length
                value = command[value_start_index:value_start_index+value_end_index]
                par_values[switch] = value
            else:
                par_values[switch] = ""

        return par_values
