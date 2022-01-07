# AutoPEEPTitration

## Overview

This project investigates the effects of different Machine Learning approaches and Neural Networks on the prediction of optimal 
compliance during PEEP ventilation.

## Prerequisites

This programme can be run without a GPU and has been successfully tested on average CPUs (Intel i5). 
Among others, the following libraries were used:

- python 3.7.9
- numpy 1.19
- pandas 1.2
- pytorch 1.6
- scipy 1.5

# Important Building Blocks

## (main.py) structure

All parts of the programme can be started by executing main.py. To use individual networks or ML algorithms, these must 
be set in config.py. All other settings (data used, splits, plots, hyperparameters etc.) can also be set there. No settings 
should be made elsewhere in the code. All parameters to be set are explained briefly in config.py. Furthermore, all 
important functions are briefly described and/or meaningfully named.

## Dataloader.py

To load data use the function loader() anywhere in the code. It gives back the loaded data and the target vector. 
Dataloading consists of 4 basic steps.

- The data is loaded from the provided .csv files in a given folder.
- The data is truncated and standardised with a selectable number of breaths per PEEP step.
- The gradient of the compliance is calculated for each PEEP step.
- A target vector is created using one-hot encoding, where a 1 represents the records belonging to the optimal PEEP step 
(in terms of compliance).

## Neural Network (SimpleNN.py)

The network can be called via the function neuralnetwork(data, target) and forms a self-contained programme sequence for calculation. 
Only the data must be handed over and all necessary preferences must be determined in config.py. The function maps the 
entire life cycle. The data is divided (train, val, test), the training is carried out and evaluated, and the network is evaluated.
Layers and knots can be changed as desired, but are sensibly chosen in this version. Furthermore, all weights of the 
network are saved in the file model.pth after successful training in order to be able to access them again later. 
This makes it possible to evaluate and use the network externally. If requested, the results can be saved externally in 
a file (classification_report.txt).

## Machine Learning approaches

To compare the results of the NN, we tried and evaluated different classical ML approaches. The file Compare_ML_Classifiers.py 
compares the approaches kNN, Random Forest, Naive Bayes and SVM. Furthermore, it is possible to visualise the results. 
More precise implementations, including different dimensions of the input data, as well as further visualisation options 
were realised for the kNN and SVM approaches (KNNmodel.py, SVMmodel.py).

## Output

model.pth  
Saved version of the calculated weights of the network. Can be used for further tests outside this programme or loaded 
again for testing via config.py.

classification_report.txt  
Results of the network. Saved text file with the focus on the F1 score.

Plots  
Various different plots of the data and results are possible through simple settings in config.py. These include loss 
curves, data plots and results from ML approaches.