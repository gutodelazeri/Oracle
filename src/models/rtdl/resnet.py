import zero
import numpy as np
import src.models.rtdl.lib as lib
import src.models.rtdl.bin.evaluation as evaluation
from pandas import isna
from src.models.rtdl.bin.resnet import *
from src.parsing.constants import RegressionModelSetup
from src.models.regression_model import RegressionModel


class DLresnet(RegressionModel):

    def __init__(self, parameters_info, model_info):
        acotsp_categoricals = ["instance", "algorithm", "localsearch", "dlb"]
        acotsp_numericals = ["alpha", "beta", "rho", "ants", "q0", "rasrank", "elitistants", "nnls"]
        lkh_categoricals = ["instance", "BACKTRACKING", "CANDIDATE_SET_TYPE", "EXTRA_CANDIDATE_SET_TYPE", "GAIN23",
                            "GAIN_CRITERION", "INITIAL_TOUR_ALGORITHM", "RESTRICTED_SEARCH", "SUBGRADIENT",
                            "SUBSEQUENT_PATCHING"]
        lkh_numericals = ["ASCENT_CANDIDATES", "BACKBONE_TRIALS", "EXTRA_CANDIDATES", "INITIAL_STEP_SIZE", "KICK_TYPE",
                          "KICKS", "MAX_CANDIDATES", "MOVE_TYPE", "PATCHING_A", "PATCHING_C", "POPULATION_SIZE",
                          "SUBSEQUENT_MOVE_TYPE"]

        args, output = lib.load_config(["--evaluate", model_info.model_parameters["weights"]])

        zero.set_randomness(args["seed"])

        self.tf = evaluation.Normalization(args, args["seed"])

        if model_info.model_parameters["scenario"] == "acotsp":
            self.numericals = acotsp_numericals
            self.categoricals = acotsp_categoricals
        elif model_info.model_parameters["scenario"] == "lkh":
            self.numericals = lkh_numericals
            self.categoricals = lkh_categoricals
        else:
            print(f"Unknown scenario: {model_info.model_parameters['scenario']}")
            exit(1)

        self.device = lib.get_device()
        self.model = ResNet(d_numerical=self.tf.n_num_features, categories=self.tf.cat_values,
                            d_out=1,  ## regression hardcoded
                            **args['model'],
                            ).to(self.device)
        self.model.eval()
        checkpoint_path = output / 'checkpoint.pt'
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])

    def fit(self, X, y):
        pass

    def predict(self, values):
        num = []
        cat = []
        for par in self.numericals:
            if not isna(values[par]):
                num.append(float(values[par]))
            else:
                num.append(np.nan)

        for par in self.categoricals:
            if not isna(values[par]):
                cat.append(values[par])
            else:
                cat.append('nan')

        x_num = np.array([num]).reshape(1, -1)
        x_cat = cat
        x_num, x_cat = self.tf.normalize_x(x_num, x_cat)

        with torch.no_grad():
            if self.device.type != 'cpu':
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
            y_raw = self.model(x_num, x_cat)

        return float(self.tf.normalize_y(y_raw))


if __name__ == "__main__":
    m = RegressionModelSetup(model={}, model_parameters={"seed": 0,
                                                         "weights": "src/models/rtdl/output/acotsp-30K/resnet/tuned/13.toml",
                                                         "scenario": "acotsp"}, preprocessing={})

    rn = DLresnet({}, m)

    d = {"instance": "121", "algorithm": "eas", "localsearch": "2",
         "alpha": 3.24, "beta": 4.74, "rho": 0.63, "ants": 75, "q0": np.nan,
         "rasrank": np.nan, "elitistants": 683, "nnls": 50, "dlb": 1}

    print(rn.predict(d))
