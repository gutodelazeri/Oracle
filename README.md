# CSMTOA

This repository contains the reference implementation of the tool used in the paper
"Comparing surrogate models for tuning optimization algorithms". The following sections
explain how to set up your environment, use the tool and extend it with other regression models.

- [1. Preparing the environment](#1-preparing-the-environment)
- [2. Usage](#2-usage)
  - [2.1. The configuration file](#21-the-configuration-file)
- [3. Adding new models](#3-adding-new-models)
- [4. How to cite](#4-how-to-cite)

## 1. Preparing the environment
This tutorial assumes that you have Conda installed in your machine
and accessible through the command line. If that is the case, type the following commands
in your terminal to clone download this repository and set up the environment. 

```bash
git clone <repository url>
cd oracle 
conda env create -f environment.yml
conda activate oracle
pip install -e .
```

## 2. Usage
To use this tool together with a configurator, two steps are necessary. 
Suppose you want to run iRace on a surrogate model of the ACOTSP algorithm.
First, you have to start the server. Inside the root of this project,
type the following.

```bash
cp examples/catboost.toml .
python3 src/oracle.py catboost.toml
```
This will start a server that, given a valid configuration in the parameter
space defined in scenarios/acotsp.toml, will use Catboost trained on data/acotsp.csv
to make a prediction of the objective value obtained with this configuration (here a configuration is
an instance together with the hyperparameters of ACOTSP).

The second step is to modify the target algorithm to call ./scripts/stub.py instead of
the actual algorithm. A call to ./scripts/stub.py would look something like this:
```bash
python3 scripts/stub.py -instance 121 -algorithm eas -localsearch 2 -alpha 3.24 -beta 4.74 -rho 0.63 -ants 75 -elitistants 683 -nnls 50 -dlb 1
```
The call will print the predicted value in the standard output.

### 2.1. The configuration file
The configuration file is the only file passed as argument to
oracle.py. It contains information about how to set up the server,
the parameter space, the regression model that will be used, and the 
training data. Some concrete examples of valid configuration files 
can be found in the folder ./examples.

The configuration file is a .toml consisting of five sections. Below, we give
an overview of each section.

#### general
- **scenario_file**: a path to a .toml file defining the parameter space. Check the folder ./scenarios for concrete examples.
- **port**: the port behind which ./src/oracle.py will wait for requests.

#### model
Every model must be a class that implements the interface
defined in ./src/models/regression_model.py. To load a model, it is
necessary to specify the path of the module inside which the model is implemented
and the name of class that implements the interface.

- **model_path**: path (relative to the package src) to the module implementing the model (e.g., "models.base.RFranger")
- **model_name**: name of the class inside "model_file" that implements the required interface (e.g., "RFranger").

#### model_parameters
In this section you can specify the hyperparameters of the regression
model (like p and k, as described in examples/interpolation.toml). Hence, the
contents of this section depend on the model that is being used.

#### preprocessing
Some models require the application of preprocessing steps
in the training data, like the imputation of missing values and 
the encoding of categorical parameters. These preprocessing steps
are specified here. In the same way as in model_parameters, the contents
of this section depend on the model being used

#### data
In this section we specify where to find the training data and 
how it should be interpreted.
- **dataset_path**: path to the file storing the training data.
- **separator**: the string or character used to separate values.  
- **labels_column**: the name of the column holding the label of each data point.

## 3. Adding new models
Using this tool with a custom regression model is really straightforward. The steps
are better illustrated with an example. Let's create a simple model that always predicts the value 10.

Inside ./src/models, create a file called my_model.py and copy and paste inside it the code below.

```python
from src.models.regression_model import RegressionModel

class MyModel(RegressionModel):
    def __init__(self, parameters_info, model_info):
        pass
    def fit(self, X, y):
        pass
    def predict(self, values):
        return 10
```
Now, create a new configuration file called config.toml with the following content.

```toml
[general]
scenario_file = "scenarios/acotsp.toml"
port = "5555"

[model]
model_path = "models.my_model"
model_name = "MyModel"

[model_parameters]

[preprocessing]

[data]
dataset_path = "data/acotsp.csv"
separator = " "
labels_column = "val"

```
At last, start the model as described in section 2.

```bash
python3 src/oracle.py config.toml
```

Any call to ./scripts/stub.py, like the one below, should return the value 10.
```bash
python3 scripts/stub.py -instance 121 -algorithm eas -localsearch 2 -alpha 3.24 -beta 4.74 -rho 0.63 -ants 75 -elitistants 683 -nnls 50 -dlb 1
```

## 4. How to cite
TBD.

