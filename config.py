# Define configs here
import os

# System

model = "SimpleNN"      #SimpleNN, Philip, kNN, SVM

epoch_number = (3000 * 10) # number must be  X % 10 = 0

learning_rate = 0.01

# define where the main program is saved
link = "C:/Users/"+str(os.getlogin())+"/PycharmProjects/AutoPEEPTitration/main.py"

# Cross-Validation: Choose between 7 different splits (Training/Test/Validation)
# With all 7 splits every data-set is used for testing (exactly 1 time)
split = 1 # Choose a number between 1-7


# IMPORTANT: Only test your network with the same split used for training (otherwise an error will occur)
# (if False, the test with the correct split is executed automatically)
test = False

#----------------------------------------------------#

# User

# how many breathes per peep step (10,20,40)
length = 20
# calculate the gradient and use it
grad = True
# print data plots of all input peep-data-sets
print = False
# plot curve of loss (--> overfitting)
loss_plot = False
# save classification report as text file
report = True
# compare ML models with each other
compare = False
