# fomlads-group-project 
# Model Comparison on the Mobile Pricing Classification Dataset
In this repository we will investigate the patterns of this dataset. The 4 models we will use:

- Fisher's Linear Discriminant Analysis
- Logistic Regression
- Random Forest Classifier
- KNN


The 4 classes in this task Price ranges in the dataset and are the following:
- 0 (low cost)
- 1 (medium cost)
- 2 (high cost)
- 3 (very high cost)

## Functionality of the files
In this folder, we have 9 Python files, all handling different aspects of the enitre project:
- data.py -The preprocessing of the dataset and a correlation heatmap function for data analysis of the dataset
- modelconstruct.py - Contains 2 functions, one for the general train-test splitting of the dataset and another for the standardisation of the dataset
- metrics.py - Has 5 functions setting out the 5 metrics that are explored in this project (Accuracy, confusion matrix, precision, recall and F1 Score), with some functions used again further down the line to calculate another metric (e.g precision and recall for F1 score)
- SoftmaxRegression.py, LDA.py, Knn.py and RandomForest.py - These contain the construction of the models investigated in this project in their respective files, they ran with their optimal hyperparameters
- plots.py - Holds 4 functions for hyperparameter analysis of 2 models (Random Forest and KNN) and their error functions which are then plotted
- main.py - Takes all the above together to provide a succinct overview of the project, runs all the models and displays their respective analysis graphs discussed in the Group Report. Ran upon a command line command stated below in this file

Note there is a plots folder too, which contains all the plots created upon running the code (these plots mainly come from the plot.py and main.py files).

## Libraries
In this project we have used the following Python libraries:
- Numpy (For mathematical calculations for the data)
- Matplotlib (For visualisation of the dataset)
- Seaborn (Additional visualisation)
- Pandas (For helping to manipulate the dataset in a reliable way)
- Time (For duration of model calculations)
- sklearn (For the 2 models outside of lectures)
- Scipy (For construction of one-hot vectors for Softmax Regression)
- sys (For the main Python file to run the experiments)
- warnings 

## Interface instructions
To creatie the conda environment one can install via a dependencies file command:
```
conda env create -f dependencies.yml
```
This exports the relevant libraries (time, sys and warnings are already in the base Python) into an new environment called my_env_project, which can then be activated by:
```
conda activate my_env_project
```

Then, the code can be run through the main command for plots and results
```
python main.py MobilePricingUpdated.csv
```


## Reproducing the results 
For the duration of the running of the code (the main file command), it takes roughly around 3 minutes and 30 seconds, depending on your machine.
Randomness was used in this project, mainly for Random Forest, which used the parameter random_state, the random seed of 42 was used to ensure reproducible results.
































