import sys
from src.parsing.constants import *
from src.parsing.parser import Parser


class Scenario:
    def __init__(self, scenario_file: str) -> None:
        """
            :param scenario_file: a path to a .toml file containing the parsing description
        """
        parser = Parser()
        self.parameters = parser.parse_scenario_file(scenario_file)

    def get_parameters(self):
        """
            :return: a dictionary mapping parameter names to an instance of ParameterData
        """
        return self.parameters

    def get_par_names(self) -> List[str]:
        """
            :return: a list with the name of all parameters (e.g. ["alpha", "beta", "localsearch", ...])
        """
        return list(self.parameters.keys())

    def get_par_type(self, parameter_name: str) -> ParameterType:
        """
            :param parameter_name: a string specifying a parameter name (e.g., "rho")

            :return: the type of 'parameter_name' (e.g., ParameterType.real)
        """
        return self.parameters[parameter_name].type

    def switch_to_name(self, par_switch: str) -> str:
        """
            :param par_switch: a string specifying a switch of a given parameter (e.g., "--beta" for parameter "beta")

            :return: a string representing the parameter name associated with the switch 'par_switch'. (e.g., "--beta" -> "beta")
        """
        result = list(filter(lambda p: p.switch == par_switch, self.parameters.values()))
        if len(result) != 1:
            sys.stderr.write(f"There are {len(result)} sharing the same switch: {result}\n")
        else:
            return result[0].name

    def get_all_switches(self) -> List[str]:
        """
           :return: a list of all switches (e.g. ["--alpha ", "--beta ", "--localsearch ", ...])
        """
        return [p.switch for p in self.parameters.values()]
