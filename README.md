# bayesclean

#### Introduction
Source code of BClean cleaning system. The cleaning process flowchart is shown below.

![](https://github.com/thethe-github/BClean/blob/master/img/overview.png)

#### Module Introduction

1. BClean file: This is the main file of the cleaning system, which receives user-defined parameters, defines core functions, including structure generation, parameter estimation, and inference, and calls core classes of various modules.

2. Analysis file: Used to evaluate the precision, recall, and running time of the cleaning results.

3. src folder: Includes User Constraint (UC) class, Bayesian Network Structure class, Compensation Classification, Inference Strategy class, etc.

4. example folder: Contains BClean workflow code for each dataset.

5. dataset folder: Stores test datasets, including py files with added noise in real datasets.

6. baseline folder: Stores source code of comparison methods and their papers.

#### User Constraint (UC) Writing Method
1. Initialize UC:

uc = UC(dirty_data)

2. View the recommended regular expression for each column:

uc.PatternDiscovery()

3. Define UC for specific attributes:

uc.build(attr = "birthyear", type = "Categorical", min_v = 4, max_v = 4, null_allow = "N", repairable = "Y", pattern = re.compile(r"([1][9][6-9][0-9])")) 

4. Get the user constraints for the data:

uc.get_uc()

#### User Guide

1. Install dependencies (or in conda virtual environment):

pip install -r requirements.txt

2. Run the cleaning (or in conda virtual environment):

Taking hospital as an example, run the command: python /example/hospital.py

You can modify parameters such as infer_strategy and model_choice in /example/hospital.py.

Add a path to model_save_path in /example/hospital.py to save the model's pkl file.

