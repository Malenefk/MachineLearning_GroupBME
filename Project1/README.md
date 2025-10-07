# Project 1 - FYSSTK4155

# Authors: Birk Skogsrud, Elias Bjørgum Micaelsen-Fuglesang & Malene Fjeldsrud Karlsen

Project 1 in Machine Learning is based on creating a polynomial fit the function Runge's function. Project one includes scripts, plots and a scientific report describing the process and the results. 

The Project 1 folder is divided into several folders. Beneath is an explanation on how to navigate:
* Code
    * One folder called task1a-1f. This folder includes four notebooks running the codes found in functions.py. These are the same codes used to plot the figures found in the report. The functions.py file is a python script contianing all functions needed to reproduce the results for task 1a, 1b, 1c and 1d. A requirments text file for these specific task are also in the folder. 



One folder called task1e-1h. This folder includes scripts needed to reproduce the results for task 1e, 1f, 1g and 1h. A requirments text file for these specific task are also in the folder.

- **`GDvsSGD.ipynb`** – compares Gradient Descent (GD), Stochastic Gradient Descent (SGD), and adaptive optimizers for OLS, Ridge, and LASSO regression.  
- **`Bootstrap.ipynb`** – performs bootstrap resampling to study bias and variance when changing polynomial degree.  
- **`K-fold.ipynb`** – runs K-fold cross-validation to estimate training and test error for different model complexities.

These notebooks contain the experiments and analysis:

- **`GDvsSGD.ipynb`** – compares Gradient Descent (GD), Stochastic Gradient Descent (SGD), and adaptive optimizers for OLS, Ridge, and LASSO regression.  
- **`Bootstrap.ipynb`** – performs bootstrap resampling to study bias and variance when changing polynomial degree.  
- **`K-fold.ipynb`** – runs K-fold cross-validation to estimate training and test error for different model complexities.

### Python Module Folder: `mlmods/`

This folder acts as a small local library that all notebooks import from.  
It contains helper functions for data generation, optimization, plotting, and evaluation.

- **`__init__.py`** – marks this folder as a Python package.  
- **`closed_form.py`** – closed-form solutions for OLS and Ridge regression.  
- **`data_utils.py`** – functions for data creation (Runge function), polynomial features, and standardization.  
- **`grads.py`** – gradient and subgradient functions for OLS, Ridge, and LASSO regression.  
- **`metrics_utils.py`** – basic metrics such as mean squared error (MSE).  
- **`optims.py`** – optimization algorithms including GD, Momentum, AdaGrad, RMSprop, and Adam, plus SGD variants.  
- **`plotting_utils.py`** – plotting helpers for convergence curves and coefficient comparisons.  
- **`resampling_utils.py`** – utilities for creating K-fold splits.
- **`trainers.py`** – functions that train and compare different optimizer families.

* LLM
    * Contains information on how we have used ChatGPT in the project. 

* Report
    * Contains a PDF of our finalized scientific report. 
