import zmq
import sys
from src.parsing.parser import Parser
from src.scenario import Scenario
from src.parsing.constants import ConfigKeywords
from predictor import Predictor
from typing import Dict


class Oracle:
    def __init__(self, config_file: str) -> None:
        """
            :param config_file: a path to the .toml file
        """

        self.config_file = config_file
        self.setup = None
        self.model = None
        self.training = None

        self.parser = Parser()
        self.setup, self.model, self.training = self.parser.parse_config_file(config_file)
        self.scenario = Scenario(self.setup.scenario_file)

        self.predictor = Predictor(scenario=self.scenario,
                                   model_description=self.model,
                                   training_data=self.training)

    def print_info(self) -> None:
        """
            Print some information about the Oracle
        """

        print("---------- ORACLE ----------")
        print(f"     Scenario file: {self.setup.scenario_file}")
        print(f"     Model: {self.model.model[ConfigKeywords.model_name]}")
        for k, v in self.model.model_parameters.items():
            print(f"        {k}: {v}")
        print(f"    Data: {self.training.dataset_path}")
        print(f"        separator: '{self.training.separator}'")
        print(f"    Port: {self.setup.port_number}")
        print("----------------------------")

    def parse_input_string(self, input_string: str) -> Dict[str, str]:
        """
            :param input_string: a string describing a configuration (e.g., "--alpha 0.1 --beta 0.4 ...")

            :return: a dictionary mapping parameter names to its values  (e.g., {"alpha": 0.1, "beta": 0.4, ...}).
                    Missing parameters receive the value np.nan.
        """

        mapping = self.parser.parse_command_line_argument(input_string, self.scenario.get_all_switches())
        output = dict()
        for switch in mapping:
            par_name = self.scenario.switch_to_name(switch)
            output[par_name] = mapping[switch]

        return output

    def guess_value(self, serial_data: bytes) -> bytes:
        """
            :param serial_data: a string representing a configuration
        
            :return: an encoded string representing the guessed value.
        """

        raw_input = serial_data.decode("utf-8")

        parameters_dict = self.parse_input_string(raw_input)

        result = str(self.predictor.predict(parameters_dict)).encode("utf-8")

        return result


if __name__ == "__main__":
    print("Loading...")

    if len(sys.argv) != 2:
        print("Usage: python3 src.py config_file")
        exit(1)
    else:
        args = sys.argv[1]

    oracle = Oracle(config_file=sys.argv[1])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{oracle.setup.port_number}")

    oracle.print_info()

    while True:
        message = socket.recv()
        val = oracle.guess_value(message)
        socket.send(val)
